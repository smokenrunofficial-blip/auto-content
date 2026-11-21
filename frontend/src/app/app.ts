import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Component, DestroyRef, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
// import { finalize, switchMap, takeWhile, timer } from 'rxjs';
import { finalize, Observable, switchMap, takeWhile, timer } from 'rxjs';

interface PipelineVideo {
  id: string;
  title: string;
  url: string;
  license: string;
  source_preview_url: string;
}

interface PipelineStage {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
}

interface ProcessingJob {
  id: string;
  keyword: string;
  videoId?: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  progress: number;
  message: string;
  stages: PipelineStage[];
  video?: PipelineVideo;
  result?: {
    output_file: string;
    script: string;
    scene_summary: string;
    duration_seconds: number;
    preview_url: string;
    download_url: string;
  };
  error?: string;
}

interface YouTubeAuthStatus {
  connected: boolean;
  cookieCount: number;
}

const youtubeAuthRequiredMessage = [
  'YouTube richiede l’accesso o la conferma dell’età.',
  'Clicca su "Login YouTube (Cache)" per importare la sessione dal browser o caricare cookies.txt.',
].join(' ');

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  private readonly http = inject(HttpClient);
  private readonly destroyRef = inject(DestroyRef);

  protected keyword = 'urban gardening';
  protected brief = 'Un dettaglio sorprendente che vale la pena conoscere';
  protected rightsConfirmed = false;
  protected readonly loading = signal(false);
  protected readonly error = signal('');
  protected readonly pipelineJob = signal<ProcessingJob | null>(null);
  protected readonly youtubeAuthPanelOpen = signal(false);
  protected readonly youtubeAuthBusy = signal(false);
  protected readonly youtubeAuthChecked = signal(false);
  protected readonly youtubeAuthConnected = signal(false);
  protected readonly youtubeAuthMessage = signal('');
  protected readonly youtubeAuthError = signal('');
  protected youtubeBrowser = 'chrome';
  protected readonly stageLabels: Record<string, string> = {
    search: 'Ricerca YouTube e verifica licenze',
    download: 'Video scaricato',
    frame_analysis: 'Analisi visiva locale',
    source_transcription: 'Trascrizione video · Whisper',
    story_generation: 'Script narrativo',
    // elevenlabs_synthesis: 'Voce narrante · ElevenLabs',
    elevenlabs_synthesis: 'Sintesi voce · TTS locale',
    narration_transcription: 'Timestamp parole · Whisper',
    rendering: 'Montaggio 9:16 e sottotitoli',
  };

  constructor() {
    this.refreshYoutubeAuthStatus();
  }

  protected toggleYoutubeAuthPanel(): void {
    const opening = !this.youtubeAuthPanelOpen();
    this.youtubeAuthPanelOpen.set(opening);
    this.youtubeAuthError.set('');
    this.youtubeAuthMessage.set('');
    if (opening) this.refreshYoutubeAuthStatus();
  }

  protected importYoutubeBrowserSession(): void {
    this.saveYoutubeAuth(
      this.http.post<YouTubeAuthStatus>('/api/youtube/auth', {
        mode: 'browser',
        browser: this.youtubeBrowser,
      }),
      'Sessione YouTube importata nella cache locale.',
    );
  }

  protected async uploadYoutubeCookieFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    if (file.size > 1024 * 1024) {
      this.youtubeAuthError.set('Il file cookies.txt deve essere più piccolo di 1 MB.');
      return;
    }

    try {
      const contents = await file.text();
      this.saveYoutubeAuth(
        this.http.post<YouTubeAuthStatus>('/api/youtube/auth', contents, {
          headers: { 'Content-Type': 'text/plain' },
        }),
        'File cookies.txt salvato nella cache locale.',
      );
    } catch {
      this.youtubeAuthError.set('Non è stato possibile leggere il file cookies.txt.');
    }
  }

  protected download(): void {
    const keyword = this.keyword.trim();
    const brief = this.brief.trim();
    if (
      keyword.length < 2 ||
      brief.length < 5 ||
      !this.rightsConfirmed ||
      this.loading()
    ) {
      return;
    }

    this.loading.set(true);
    this.error.set('');
    this.pipelineJob.set(null);

    this.http
      .post<ProcessingJob>('/api/video-jobs', {
        keyword,
        brief,
        language: 'it',
        rightsConfirmed: this.rightsConfirmed,
      })
      .pipe(
        switchMap((initialJob) => {
          this.pipelineJob.set(initialJob);
          return timer(500, 1200).pipe(
            switchMap(() => this.http.get<ProcessingJob>(`/api/video-jobs/${initialJob.id}`)),
            takeWhile(
              (job) => job.status === 'queued' || job.status === 'running',
              true,
            ),
          );
        }),
        takeUntilDestroyed(this.destroyRef),
        finalize(() => this.loading.set(false)),
      )
      .subscribe({
        // next: (job) => this.pipelineJob.set(job),
        next: (job) => {
          const authMessage = this.youtubeAuthHint(job.error || job.message);
          if (job.status === 'failed' && authMessage) {
            this.error.set(authMessage);
            this.pipelineJob.set({ ...job, error: authMessage, message: authMessage });
            return;
          }
          this.pipelineJob.set(job);
        },
        error: (error: unknown) => {
          // const message =
          //   error instanceof HttpErrorResponse &&
          //   typeof error.error?.error === 'string'
          //     ? error.error.error
          //     : 'Impossibile avviare il job. Controlla API, Python e servizi richiesti.';
          const rawMessage =
            error instanceof HttpErrorResponse &&
            typeof error.error?.error === 'string'
              ? error.error.error
              : 'Impossibile avviare il job. Controlla API, Python e servizi richiesti.';
          // this.error.set(message);
          this.error.set(this.youtubeAuthHint(rawMessage) || rawMessage);
        },
      });
  }

  private refreshYoutubeAuthStatus(): void {
    this.http
      .get<YouTubeAuthStatus>('/api/youtube/auth/status')
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (status) => {
          this.youtubeAuthConnected.set(status.connected);
          this.youtubeAuthChecked.set(true);
        },
        error: () => {
          this.youtubeAuthConnected.set(false);
          this.youtubeAuthChecked.set(true);
        },
      });
  }

  private saveYoutubeAuth(
    request: Observable<YouTubeAuthStatus>,
    successMessage: string,
  ): void {
    this.youtubeAuthBusy.set(true);
    this.youtubeAuthError.set('');
    this.youtubeAuthMessage.set('');
    request
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        finalize(() => this.youtubeAuthBusy.set(false)),
      )
      .subscribe({
        next: (status) => {
          this.youtubeAuthConnected.set(status.connected);
          this.youtubeAuthChecked.set(true);
          this.youtubeAuthMessage.set(successMessage);
        },
        error: (error: unknown) => {
          const message =
            error instanceof HttpErrorResponse &&
            typeof error.error?.error === 'string'
              ? error.error.error
              : 'Non è stato possibile salvare la sessione YouTube.';
          this.youtubeAuthError.set(message);
        },
      });
  }

  private youtubeAuthHint(message: string): string | undefined {
    const needsYoutubeAuth =
      /sign in to confirm your age|confirm your age/i.test(message) ||
      /--cookies/i.test(message) ||
      /youtube.{0,80}(?:cookie|login)|(?:cookie|login).{0,80}youtube/i.test(message);
    return needsYoutubeAuth
      ? youtubeAuthRequiredMessage
      : undefined;
  }
}
