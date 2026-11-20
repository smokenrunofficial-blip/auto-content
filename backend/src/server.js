import { execFile, spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { access, readdir, realpath } from 'node:fs/promises';
import { dirname, extname, isAbsolute, relative, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import express from 'express';

const backendDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const envFilePath = resolve(backendDirectory, '.env');
if (existsSync(envFilePath)) {
  process.loadEnvFile(envFilePath);
}

const app = express();
const execFileAsync = promisify(execFile);
const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const downloaderPath = resolve(projectRoot, 'scripts', 'youtube_downloader.py');
const VIDEO_CANDIDATE_LIMIT = 10;
const scriptsDirectory = resolve(projectRoot, 'scripts');
const downloadsDirectory = resolve(projectRoot, 'downloads');
const processedDirectory = resolve(downloadsDirectory, 'processed');
const pipelineStages = [
  'search',
  'download',
  'frame_analysis',
  'source_transcription',
  'story_generation',
  'elevenlabs_synthesis',
  'narration_transcription',
  'rendering',
];
const stageProgressRanges = {
  search: [0, 5],
  download: [5, 15],
  frame_analysis: [15, 28],
  source_transcription: [28, 43],
  story_generation: [43, 50],
  elevenlabs_synthesis: [50, 65],
  narration_transcription: [65, 78],
  rendering: [78, 100],
};
const allowedVideoExtensions = new Set(['.mp4', '.mkv', '.webm', '.mov', '.m4v']);
const jobs = new Map();
const port = Number(process.env.PORT ?? 3000);

app.use(express.json({ limit: '20kb' }));

function createPipelineJob(keyword) {
  return {
    id: randomUUID(),
    keyword,
    videoId: undefined,
    status: 'queued',
    stage: 'search',
    progress: 0,
    message: 'Job in coda.',
    stages: pipelineStages.map((name) => ({
      name,
      status: 'pending',
      progress: 0,
      message: '',
    })),
    video: undefined,
    result: undefined,
    outputPath: undefined,
    error: undefined,
    updatedAt: Date.now(),
    child: undefined,
  };
}

function updatePipelineProgress(job, event) {
  if (event.stage === 'completed') {
    job.stages.forEach((stage) => {
      stage.status = 'completed';
      stage.progress = 100;
    });
    job.stage = 'completed';
    job.progress = 100;
    job.message = String(event.message ?? 'Video pronto.');
    job.updatedAt = Date.now();
    return;
  }

  if (!pipelineStages.includes(event.stage)) return;
  const stageIndex = pipelineStages.indexOf(event.stage);
  job.stages.forEach((stage, index) => {
    if (index < stageIndex) {
      stage.status = 'completed';
      stage.progress = 100;
    }
  });

  const stageProgress = Math.max(0, Math.min(100, Number(event.stage_progress) || 0));
  const currentStage = job.stages[stageIndex];
  currentStage.status = stageProgress >= 100 ? 'completed' : 'running';
  currentStage.progress = stageProgress;
  currentStage.message = String(event.message ?? '');
  const [start, end] = stageProgressRanges[event.stage];
  job.stage = event.stage;
  job.progress = Math.round(start + ((end - start) * stageProgress) / 100);
  job.message = currentStage.message;
  job.updatedAt = Date.now();
}

function failPipelineJob(job, message, stageName) {
  job.status = 'failed';
  job.error = message || 'Elaborazione non riuscita.';
  job.message = 'Elaborazione non riuscita.';
  const activeStage = job.stages.find((stage) => stage.name === stageName) ??
    job.stages.find((stage) => stage.status === 'running') ??
    job.stages.find((stage) => stage.status === 'pending');
  if (activeStage) {
    activeStage.status = 'failed';
    activeStage.message = job.error;
  }
  job.updatedAt = Date.now();
}

function serializePipelineJob(job) {
  return {
    id: job.id,
    keyword: job.keyword,
    videoId: job.videoId,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    message: job.message,
    stages: job.stages,
    video: job.video
      ? {
          id: job.video.id,
          title: job.video.title,
          url: job.video.url,
          license: job.video.license,
          source_preview_url: `/api/video-jobs/${job.id}/source`,
        }
      : undefined,
    result: job.result
      ? {
          ...job.result,
          preview_url: `/api/video-jobs/${job.id}/preview`,
          download_url: `/api/video-jobs/${job.id}/download`,
        }
      : undefined,
    error: job.error,
  };
}

function runJobProcess(job, executable, args, cwd, onResult) {
  const child = spawn(executable, args, {
    cwd,
    env: {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
    },
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stdoutBuffer = '';
  let stderr = '';
  let resultReceived = false;
  let resultHandled = false;
  let processClosed = false;
  let processSucceeded = false;
  job.child = child;
  job.status = 'running';

  const completeJobIfReady = () => {
    if (
      !processClosed ||
      !processSucceeded ||
      !resultHandled ||
      job.status === 'failed' ||
      job.progress !== 100 ||
      !job.result
    ) {
      return;
    }

    job.status = 'completed';
    job.stage = 'completed';
    job.progress = 100;
    job.message = 'Video verticale pronto.';
    job.stages.forEach((stage) => {
      stage.status = 'completed';
      stage.progress = 100;
    });
    job.updatedAt = Date.now();
  };

  const handleLine = (line) => {
    if (!line.trim() || job.status === 'failed') return;
    console.log('[PY STDOUT]:', line);
    let event;
    try {
      event = JSON.parse(line);
    } catch {
      console.error('[PY JSON PARSE ERROR]:', line);
      failPipelineJob(job, 'Un processo Python ha restituito un evento non valido.', job.stage);
      child.kill();
      return;
    }

    if (event.type === 'progress') {
      updatePipelineProgress(job, event);
      return;
    }
    if (event.type === 'result' && !resultReceived) {
      resultReceived = true;
      Promise.resolve()
        .then(() => onResult(event, child))
        .then(() => {
          resultHandled = true;
          completeJobIfReady();
        })
        .catch((error) => {
          failPipelineJob(
            job,
            error instanceof Error ? error.message : 'Risultato non valido.',
            job.stage,
          );
          child.kill();
        });
      return;
    }

    failPipelineJob(job, 'Un processo Python ha restituito un evento inatteso.');
    child.kill();
  };

  child.stdout.on('data', (chunk) => {
    stdoutBuffer += chunk.toString();
    if (stdoutBuffer.length > 1024 * 1024) {
      failPipelineJob(job, 'Output del processo troppo grande.');
      child.kill();
      return;
    }
    const lines = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = lines.pop() ?? '';
    lines.forEach(handleLine);
  });
  child.stderr.on('data', (chunk) => {
  const text = chunk.toString();
  stderr = `${stderr}${text}`.slice(-8000);
  // console.error('[PY STDERR]:', text.trim());
  });
  child.once('error', (error) => failPipelineJob(job, error.message));
  child.once('close', (code) => {
    if (job.child !== child) return;
    if (job.status === 'failed') return;
    if (stdoutBuffer.trim()) handleLine(stdoutBuffer);
    if (code !== 0) {
      failPipelineJob(job, stderr.trim() || `Il processo Python è terminato con codice ${code}.`);
      return;
    }
    if (!resultReceived) {
      failPipelineJob(job, 'Il processo Python è terminato senza restituire un risultato.');
      return;
    }

    // if (job.progress === 100 && job.result) {
    //   job.status = 'completed';
    //   job.stage = 'completed';
    //   job.progress = 100;
    //   job.message = 'Video verticale pronto.';
    //   job.stages.forEach((stage) => {
    //     stage.status = 'completed';
    //     stage.progress = 100;
    //   });
    //   job.updatedAt = Date.now();
    // }
    processClosed = true;
    processSucceeded = true;
    completeJobIfReady();
  });
}

async function findDownloadedVideo(videoId, reportedFile) {
  if (typeof reportedFile === 'string' && reportedFile.trim()) {
    const candidatePath = resolve(projectRoot, reportedFile);
    const candidate = await realpath(candidatePath);
    const relativeCandidate = relative(downloadsDirectory, candidate);
    if (
      !relativeCandidate.startsWith('..') &&
      !isAbsolute(relativeCandidate) &&
      candidate.startsWith(resolve(downloadsDirectory, `${videoId}.`)) &&
      allowedVideoExtensions.has(extname(candidate).toLowerCase())
    ) {
      return candidate;
    }
    throw new Error('Il downloader ha restituito un percorso video non valido.');
  }

  const matchingFiles = (await readdir(downloadsDirectory, { withFileTypes: true }))
    .filter(
      (entry) =>
        entry.isFile() &&
        entry.name.startsWith(`${videoId}.`) &&
        allowedVideoExtensions.has(extname(entry.name).toLowerCase()),
    )
    .map((entry) => entry.name);
  if (matchingFiles.length !== 1) {
    throw new Error('Il file video scaricato non è stato trovato o è ambiguo.');
  }

  const inputPath = await realpath(resolve(downloadsDirectory, matchingFiles[0]));
  const relativeInput = relative(downloadsDirectory, inputPath);
  if (relativeInput.startsWith('..') || isAbsolute(relativeInput)) {
    throw new Error('Percorso del video non valido.');
  }
  return inputPath;
}

// function startVideoProcessing(job, videoId, inputPath, brief, language) {
function startVideoProcessing(job, videoId, inputPath, brief, language, onCandidateRejected) {
  job.videoId = videoId;
  job.message = 'Video scaricato; avvio analisi e montaggio.';
  runJobProcess(
    job,
    process.env.PYTHON_EXECUTABLE || 'python',
    [
      '-m',
      'video_pipeline.process_video',
      '--input',
      inputPath,
      '--video-id',
      videoId,
      '--brief',
      brief,
      '--language',
      language,
      '--rights-confirmed',
      '--allow-candidate-rejection',
    ],
    scriptsDirectory,
    async (event, child) => {
      if (event.candidate_rejected === true) {
        await onCandidateRejected(
          String(event.reason ?? 'Il video non soddisfa i requisiti visivi.'),
        );
        return;
      }
      if (typeof event.output_file !== 'string') {
        throw new Error('Il processo video non ha restituito il percorso del risultato.');
      }
      const outputPath = resolve(projectRoot, event.output_file);
      const relativeOutput = relative(processedDirectory, outputPath);
      if (!relativeOutput || relativeOutput.startsWith('..') || isAbsolute(relativeOutput)) {
        throw new Error('Il processo ha restituito un percorso di output non valido.');
      }
      job.outputPath = outputPath;
      job.result = {
        output_file: event.output_file,
        script: String(event.script ?? ''),
        scene_summary: String(event.scene_summary ?? ''),
        duration_seconds: Number(event.duration_seconds) || 0,
        clips: Array.isArray(event.clips) ? event.clips : [],
      };
    },
  );
}

function startVideoDownload(job, brief, language) {
  runJobProcess(
    job,
    process.env.PYTHON_EXECUTABLE || 'python',
    // [downloaderPath, '--keyword', job.keyword, '--limit', '1', '--progress'],
    // [downloaderPath, '--keyword', job.keyword, '--limit', '3', '--progress'],
    [
      downloaderPath,
      '--keyword',
      job.keyword,
      '--limit',
      String(VIDEO_CANDIDATE_LIMIT),
      '--progress',
    ],
    projectRoot,
    async (event) => {
      if (!Array.isArray(event.videos) || event.videos.length === 0) {
        throw new Error('Nessun video trovato per la parola chiave.');
      }

      // const video = event.videos[0];
      // if (
      //   !video ||
      //   typeof video.id !== 'string' ||
      //   !/^[A-Za-z0-9_-]{6,32}$/.test(video.id) ||
      //   typeof video.title !== 'string' ||
      //   typeof video.url !== 'string'
      //   // typeof video.license !== 'string' ||
      //   // !video.license
      //   //   .toLowerCase()
      //   //   .startsWith('creative commons attribution license')
      // ) {
      //   throw new Error('Il downloader non ha restituito un risultato valido.');
      // }
      //
      // const inputPath = await findDownloadedVideo(video.id, video.file);
      // job.videoId = video.id;
      // job.video = {
      //   id: video.id,
      //   title: video.title,
      //   url: video.url,
      //   license: video.license ?? 'None',
      // };
      // job.sourcePath = inputPath;
      // startVideoProcessing(job, video.id, inputPath, brief, language);
      let candidateIndex = 0;
      const tryNextCandidate = async (rejectionReason) => {
        if (rejectionReason) {
          console.warn(
            `Candidato ${job.videoId} rifiutato; provo il successivo: ${rejectionReason}`,
          );
        }
        if (candidateIndex >= event.videos.length) {
          failPipelineJob(
            job,
            rejectionReason || 'Nessun candidato contiene clip animate pulite sufficienti.',
            'frame_analysis',
          );
          return;
        }

        const attemptNumber = candidateIndex + 1;
        const video = event.videos[candidateIndex++];
        if (
          !video ||
          typeof video.id !== 'string' ||
          !/^[A-Za-z0-9_-]{6,32}$/.test(video.id) ||
          typeof video.title !== 'string' ||
          typeof video.url !== 'string'
          // typeof video.license !== 'string' ||
          // !video.license
          //   .toLowerCase()
          //   .startsWith('creative commons attribution license')
        ) {
          // throw new Error('Il downloader non ha restituito un risultato valido.');
          await tryNextCandidate(
            `Il candidato ${attemptNumber} non contiene metadati validi.`,
          );
          return;
        }

        // const inputPath = await findDownloadedVideo(video.id, video.file);
        let inputPath;
        try {
          inputPath = await findDownloadedVideo(video.id, video.file);
        } catch (error) {
          const reason = error instanceof Error ? error.message : 'file video non valido';
          await tryNextCandidate(
            `Il file del candidato ${video.id} non è utilizzabile: ${reason}`,
          );
          return;
        }
        job.videoId = video.id;
        job.video = {
          id: video.id,
          title: video.title,
          url: video.url,
          license: video.license ?? 'None',
        };
        job.sourcePath = inputPath;
        startVideoProcessing(
          job,
          video.id,
          inputPath,
          brief,
          language,
          tryNextCandidate,
        );
      };

      await tryNextCandidate('');
    },
  );
}

app.get('/api/health', (_request, response) => {
  response.json({ status: 'ok' });
});

app.post('/api/downloads', async (request, response) => {
  const { keyword, limit = 2 } = request.body ?? {};

  if (
    typeof keyword !== 'string' ||
    keyword.trim().length < 2 ||
    keyword.trim().length > 120 ||
    !Number.isInteger(limit) ||
    limit < 1 ||
    limit > 3
  ) {
    response.status(400).json({
      error: 'Inserisci una parola chiave di 2-120 caratteri e un limite tra 1 e 3.',
    });
    return;
  }

  try {
    const { stdout } = await execFileAsync(
      process.env.PYTHON_EXECUTABLE || 'python',
      [downloaderPath, '--keyword', keyword.trim(), '--limit', String(limit)],
      {
        cwd: projectRoot,
        timeout: 10 * 60 * 1000,
        maxBuffer: 1024 * 1024,
        windowsHide: true,
      },
    );

    response.json(JSON.parse(stdout));
  } catch (error) {
    const details =
      error && typeof error === 'object' && 'stderr' in error
        ? String(error.stderr).trim()
        : '';
    const message = details || (error instanceof Error ? error.message : 'Errore inatteso.');
    console.error('Download non riuscito:', message);
    response.status(502).json({ error: message });
  }
});

app.post('/api/video-jobs', (request, response) => {
  const { keyword, brief, language = 'it', rightsConfirmed } = request.body ?? {};

  if (
    typeof keyword !== 'string' ||
    keyword.trim().length < 2 ||
    keyword.trim().length > 120 ||
    typeof brief !== 'string' ||
    brief.trim().length < 5 ||
    brief.trim().length > 400 ||
    !['de', 'en', 'es', 'fr', 'it', 'pt'].includes(language) ||
    rightsConfirmed !== true
  ) {
    response.status(400).json({
      error: 'Parola chiave, spunto narrativo, lingua o conferma dei diritti non validi.',
    });
    return;
  }

  // if (!process.env.ELEVENLABS_API_KEY || !process.env.ELEVENLABS_VOICE_ID) {
  //   response.status(503).json({
  //     error: 'Configura ELEVENLABS_API_KEY e ELEVENLABS_VOICE_ID prima di avviare il job.',
  //   });
  //   return;
  // }

  const activeJobs = [...jobs.values()].filter(
    (job) => job.status === 'queued' || job.status === 'running',
  ).length;
  if (activeJobs >= 2) {
    response.status(429).json({ error: 'Sono già attive due elaborazioni video.' });
    return;
  }

  const now = Date.now();
  for (const [id, job] of jobs) {
    if (
      (job.status === 'completed' || job.status === 'failed') &&
      now - job.updatedAt > 12 * 60 * 60 * 1000
    ) {
      jobs.delete(id);
    }
  }

  const job = createPipelineJob(keyword.trim());
  jobs.set(job.id, job);
  startVideoDownload(job, brief.trim(), language);
  response.status(202).json(serializePipelineJob(job));
});

app.get('/api/video-jobs/:id', (request, response) => {
  const job = jobs.get(request.params.id);
  if (!job || !job.keyword) {
    response.status(404).json({ error: 'Job non trovato o scaduto.' });
    return;
  }
  response.json(serializePipelineJob(job));
});

app.get('/api/video-jobs/:id/source', (request, response) => {
  const job = jobs.get(request.params.id);
  if (!job || !job.sourcePath) {
    response.status(404).json({ error: 'Anteprima sorgente non ancora disponibile.' });
    return;
  }
  sendInlineVideo(response, job.sourcePath);
});

app.get('/api/video-jobs/:id/preview', (request, response) => {
  const job = jobs.get(request.params.id);
  if (!job || job.status !== 'completed' || !job.outputPath) {
    response.status(404).json({ error: 'Anteprima montata non ancora disponibile.' });
    return;
  }
  sendInlineVideo(response, job.outputPath);
});

app.get('/api/video-jobs/:id/download', async (request, response) => {
  const job = jobs.get(request.params.id);
  if (!job || job.status !== 'completed' || !job.outputPath) {
    response.status(404).json({ error: 'Output non disponibile.' });
    return;
  }

  try {
    await access(job.outputPath);
    response.download(job.outputPath, `short-${job.videoId}.mp4`, (error) => {
      if (error) {
        console.error('Download output non riuscito:', error.message);
        if (!response.headersSent) {
          response.status(500).json({ error: 'Non è stato possibile inviare il file.' });
        }
      }
    });
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
      response.status(404).json({ error: 'File di output non trovato.' });
      return;
    }
    console.error('Verifica output non riuscita:', error);
    response.status(500).json({ error: 'Non è stato possibile verificare il file di output.' });
  }
});

function sendInlineVideo(response, filePath) {
  response.sendFile(
    filePath,
    {
      headers: {
        'Content-Disposition': 'inline',
        'Cache-Control': 'no-store',
      },
    },
    (error) => {
      if (!error) return;
      if (!response.headersSent) {
        response.status(error.statusCode === 404 ? 404 : 500).json({
          error: 'Non è stato possibile caricare l’anteprima video.',
        });
      }
    },
  );
}

app.post('/api/video-processing', async (request, response) => {
  const { videoId, brief, language = 'it', rightsConfirmed } = request.body ?? {};

  if (
    typeof videoId !== 'string' ||
    !/^[A-Za-z0-9_-]{6,32}$/.test(videoId) ||
    typeof brief !== 'string' ||
    brief.trim().length < 5 ||
    brief.trim().length > 400 ||
    !['de', 'en', 'es', 'fr', 'it', 'pt'].includes(language) ||
    rightsConfirmed !== true
  ) {
    response.status(400).json({
      error: 'ID, spunto narrativo, lingua o conferma dei diritti non validi.',
    });
    return;
  }

  // if (!process.env.ELEVENLABS_API_KEY || !process.env.ELEVENLABS_VOICE_ID) {
  //   response.status(503).json({
  //     error: 'Configura ELEVENLABS_API_KEY e ELEVENLABS_VOICE_ID prima di avviare il montaggio.',
  //   });
  //   return;
  // }

  const activeJobs = [...jobs.values()].filter(
    (job) => job.status === 'queued' || job.status === 'running',
  ).length;
  if (activeJobs >= 2) {
    response.status(429).json({ error: 'Sono già attive due elaborazioni video.' });
    return;
  }

  let matchingFiles;
  try {
    matchingFiles = (await readdir(downloadsDirectory, { withFileTypes: true }))
      .filter(
        (entry) =>
          entry.isFile() &&
          entry.name.startsWith(`${videoId}.`) &&
          allowedVideoExtensions.has(extname(entry.name).toLowerCase()),
      )
      .map((entry) => entry.name);
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
      response.status(404).json({ error: 'Cartella downloads/ non trovata.' });
      return;
    }
    throw error;
  }

  if (matchingFiles.length !== 1) {
    response.status(404).json({ error: 'Video scaricato non trovato o ambiguo.' });
    return;
  }

  const inputPath = await realpath(resolve(downloadsDirectory, matchingFiles[0]));
  const relativeInput = relative(downloadsDirectory, inputPath);
  if (relativeInput.startsWith('..') || isAbsolute(relativeInput)) {
    response.status(400).json({ error: 'Percorso del video non valido.' });
    return;
  }

  for (const [id, job] of jobs) {
    if (
      (job.status === 'completed' || job.status === 'failed') &&
      Date.now() - job.updatedAt > 12 * 60 * 60 * 1000
    ) {
      jobs.delete(id);
    }
  }

  const id = randomUUID();
  const job = {
    id,
    videoId,
    status: 'queued',
    stage: 'download',
    progress: 3,
    message: 'Video già scaricato: file locale verificato.',
    stages: pipelineStages.map((name) => ({
      name,
      status: name === 'download' ? 'completed' : 'pending',
      progress: name === 'download' ? 100 : 0,
      message: name === 'download' ? 'Video locale verificato.' : '',
    })),
    result: undefined,
    outputPath: undefined,
    error: undefined,
    updatedAt: Date.now(),
    stdoutBuffer: '',
    stderr: '',
    child: undefined,
  };
  jobs.set(id, job);

  const child = spawn(
    process.env.PYTHON_EXECUTABLE || 'python',
    [
      '-m',
      'video_pipeline.process_video',
      '--input',
      inputPath,
      '--video-id',
      videoId,
      '--brief',
      brief.trim(),
      '--language',
      language,
      '--rights-confirmed',
    ],
    {
      cwd: scriptsDirectory,
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
      },
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  job.child = child;
  job.status = 'running';

  const failJob = (message) => {
    job.status = 'failed';
    job.error = message || 'Elaborazione non riuscita.';
    job.message = 'Elaborazione non riuscita.';
    const activeStage = job.stages.find((stage) => stage.status === 'running');
    if (activeStage) {
      activeStage.status = 'failed';
      activeStage.message = job.error;
    }
    job.updatedAt = Date.now();
  };

  const handleLine = (line) => {
    if (!line.trim()) return;
    let event;
    try {
      event = JSON.parse(line);
    } catch {
      failJob('Il processo Python ha restituito un evento non valido.');
      child.kill();
      return;
    }

    if (event.type === 'progress' && pipelineStages.includes(event.stage)) {
      const currentIndex = pipelineStages.indexOf(event.stage);
      job.stages.forEach((stage, index) => {
        if (index < currentIndex) {
          stage.status = 'completed';
          stage.progress = 100;
        }
      });
      const currentStage = job.stages[currentIndex];
      currentStage.status = event.stage_progress >= 100 ? 'completed' : 'running';
      currentStage.progress = Math.max(0, Math.min(100, Number(event.stage_progress) || 0));
      currentStage.message = String(event.message ?? '');
      job.stage = event.stage;
      job.progress = Math.max(0, Math.min(100, Number(event.progress) || 0));
      job.message = currentStage.message;
      job.updatedAt = Date.now();
      return;
    }

    if (event.type === 'progress' && event.stage === 'completed') {
      job.stages.forEach((stage) => {
        stage.status = 'completed';
        stage.progress = 100;
      });
      job.stage = 'completed';
      job.progress = 100;
      job.message = String(event.message ?? 'Video pronto.');
      job.updatedAt = Date.now();
      return;
    }

    if (event.type === 'result' && typeof event.output_file === 'string') {
      const outputPath = resolve(projectRoot, event.output_file);
      const relativeOutput = relative(processedDirectory, outputPath);
      if (!relativeOutput || relativeOutput.startsWith('..') || isAbsolute(relativeOutput)) {
        failJob('Il processo ha restituito un percorso di output non valido.');
        child.kill();
        return;
      }
      job.outputPath = outputPath;
      job.result = {
        output_file: event.output_file,
        script: String(event.script ?? ''),
        duration_seconds: Number(event.duration_seconds) || 0,
        clips: Array.isArray(event.clips) ? event.clips : [],
      };
      job.updatedAt = Date.now();
      return;
    }

    failJob('Il processo Python ha restituito un evento inatteso.');
    child.kill();
  };

  child.stdout.on('data', (chunk) => {
    job.stdoutBuffer += chunk.toString();
    if (job.stdoutBuffer.length > 1024 * 1024) {
      failJob('Output del processo troppo grande.');
      child.kill();
      return;
    }
    const lines = job.stdoutBuffer.split(/\r?\n/);
    job.stdoutBuffer = lines.pop() ?? '';
    lines.forEach(handleLine);
  });
  child.stderr.on('data', (chunk) => {
    job.stderr = `${job.stderr}${chunk.toString()}`.slice(-8000);
  });
  child.once('error', (error) => failJob(error.message));
  child.once('close', (code) => {
    if (job.status === 'failed') return;
    if (job.stdoutBuffer.trim()) handleLine(job.stdoutBuffer);
    if (code === 0 && job.result) {
      job.status = 'completed';
      job.stage = 'completed';
      job.progress = 100;
      job.message = 'Video verticale pronto.';
      job.stages.forEach((stage) => {
        stage.status = 'completed';
        stage.progress = 100;
      });
      job.updatedAt = Date.now();
      return;
    }
    failJob(job.stderr.trim() || `Il processo Python è terminato con codice ${code}.`);
  });

  response.status(202).json(serializeJob(job));
});

app.get('/api/video-processing/:id', (request, response) => {
  const job = jobs.get(request.params.id);
  if (!job) {
    response.status(404).json({ error: 'Job non trovato o scaduto.' });
    return;
  }
  response.json(serializeJob(job));
});

app.get('/api/video-processing/:id/download', async (request, response) => {
  const job = jobs.get(request.params.id);
  if (!job || job.status !== 'completed' || !job.outputPath) {
    response.status(404).json({ error: 'Output non disponibile.' });
    return;
  }

  try {
    await access(job.outputPath);
    response.download(job.outputPath, `short-${job.videoId}.mp4`, (error) => {
      if (error) {
        console.error('Download output non riuscito:', error.message);
        if (!response.headersSent) {
          response.status(500).json({ error: 'Non è stato possibile inviare il file.' });
        }
      }
    });
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
      response.status(404).json({ error: 'File di output non trovato.' });
      return;
    }
    console.error('Verifica output non riuscita:', error);
    response.status(500).json({ error: 'Non è stato possibile verificare il file di output.' });
  }
});

function serializeJob(job) {
  return {
    id: job.id,
    videoId: job.videoId,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    message: job.message,
    stages: job.stages,
    result: job.result
      ? {
          ...job.result,
          download_url: `/api/video-processing/${job.id}/download`,
        }
      : undefined,
    error: job.error,
  };
}

app.listen(port, '127.0.0.1', () => {
  console.log(`API disponibile su http://127.0.0.1:${port}`);
});
