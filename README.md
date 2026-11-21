# Niche Video Finder

Starter locale composto da un frontend Angular con Tailwind CSS, un'API Node.js e uno script Python basato su `yt-dlp`.

## Requisiti

- Node.js 20.19+ oppure 22.12+
- Python 3.10+

## Avvio

1. Installa le dipendenze Python: `python -m pip install -r requirements.txt`
2. Per il montaggio video installa anche: `python -m pip install -r requirements-video.txt`
3. Installa le dipendenze API: `cd backend; npm install`
4. Avvia l'API in un terminale: `npm run dev`
5. Installa le dipendenze Angular: `cd ../frontend; npm install`
6. Avvia il frontend in un altro terminale: `npm start`

Apri `http://localhost:4200`. Il proxy Angular inoltra le richieste `/api` al backend locale sulla porta 3000.

Su Windows, se il comando `python` non è disponibile, imposta `PYTHON_EXECUTABLE` con il percorso dell'interprete Python prima di avviare l'API. Il backend carica automaticamente `backend/.env` all'avvio (Node.js 20.19+ o 22.12+); le variabili d'ambiente già definite nel sistema hanno precedenza.

## Pipeline per gli Short

Installa FFmpeg e `ffprobe` e rendili disponibili nel `PATH`. Installa Ollama, avvialo su `http://127.0.0.1:11434` e scarica il modello visivo `ollama pull qwen2.5vl:3b` (modello selezionabile con `OLLAMA_VISION_MODEL`; host locale configurabile con `OLLAMA_HOST`). Avvia il servizio vocale locale con `python backend/tts/tts_server.py`; il default è `http://127.0.0.1:8000/api/tts` e può essere cambiato con `LOCAL_TTS_URL`. Il TTS principale è Coqui XTTS-v2 su CUDA; installa prima PyTorch e torchaudio CUDA 12.8 per RTX 50-series, poi `python -m pip install -r requirements-tts-gpu.txt`. Il modello XTTS viene scaricato al primo uso. Piper resta il fallback e richiede l'eseguibile e il modello italiano nella cartella `piper/` della root. Whisper è selezionabile con `WHISPER_MODEL` (predefinito `base`). La UI invia `POST /api/video-jobs`, che cerca e scarica il primo risultato idoneo, poi avvia la pipeline. Lo stato si controlla con `GET /api/video-jobs/:id`, l'anteprima sorgente con `GET /api/video-jobs/:id/source`, l'anteprima montata con `GET /api/video-jobs/:id/preview` e il file finale con `GET /api/video-jobs/:id/download`. `POST /api/video-processing` resta disponibile per elaborare un video già scaricato.

Il job automatico richiede la conferma esplicita dei diritti prima di iniziare. Cerca un solo video alla volta, verificando i metadati completi della licenza; sono scaricati solo risultati che dichiarano Creative Commons Attribution. Durante ricerca e download, yt-dlp emette gli aggiornamenti di avanzamento. Dopo il download, la pipeline estrae fotogrammi chiave e li invia al modello Qwen in esecuzione su Ollama locale per descrivere scena, azioni, momenti e punto focale; integra questa analisi con il rilevamento locale del movimento. Il modello genera uno script in italiano con gancio iniziale, basato sulla scena e sulla trascrizione Whisper, con un limite di parole calibrato sulla durata disponibile. Se il parlato supera la durata sorgente, lo script viene accorciato e sintetizzato di nuovo dal servizio Piper locale, che restituisce WAV. Whisper ne allinea i timestamp parola per parola; FFmpeg usa l'audio WAV per il montaggio e masterizza sottotitoli ASS statici. La UI consente di riprodurre sia la sorgente scaricata sia lo Short finito prima di scaricarlo. Fotogrammi e audio restano nell'elaborazione locale. I job sono in memoria e si perdono al riavvio del backend.

Il modello vocale predefinito è Riccardo (`backend/tts/piper/it_IT-riccardo-x_low.onnx` con il relativo `.onnx.json`). I sottotitoli usano i font ordinati alfabeticamente in `backend/tts/fonts`: il primo è il font base, gli altri sono alternati sulle parole chiave in maiuscolo; la posizione è a un terzo dal bordo inferiore. La pipeline sceglie casualmente una traccia `.mp3` o `.wav` da `assets/music`, la mixa al 10% con fade-out; se la cartella è vuota, il video viene montato senza musica.

L'analisi visiva richiede animazione/cartoon/anime e l'assenza verificata di testo o loghi in ogni momento selezionato. Il job scarica fino a tre candidati e li prova in sequenza: se un video non fornisce almeno due segmenti verificati, passa al successivo; il fallback può abbassare il punteggio estetico, mai i vincoli di sicurezza visiva.

Prima del montaggio, ogni finestra candidata viene ricontrollata da Qwen con fotogrammi distribuiti nel segmento e dai campioni di movimento locali; finestre con testo, stile non animato, fotogrammi neri o movimento insufficiente vengono scartate. Il render registra i percorsi della voce e della musica, controlla i livelli audio in ingresso e verifica che il file MP4 finale contenga una traccia AAC non silenziosa.

La voce è normalizzata a -16 LUFS sia con sia senza musica; il sottofondo resta al 10% e sfuma nell'ultimo secondo.

Il crop 9:16 usa Haar cascade di OpenCV per seguire i volti con smoothing tra i fotogrammi e i tagli; se OpenCV o il rilevamento non sono disponibili, mantiene un'inquadratura centrata senza interrompere il render. XTTS usa `XTTS_MODEL_NAME` (predefinito `tts_models/multilingual/multi-dataset/xtts_v2`), `XTTS_SPEED=0.92`, `XTTS_DEVICE=cuda:0` e `XTTS_MIN_FREE_VRAM_MB=5000`; `VOICE_REFERENCE_WAV` seleziona un WAV locale per il timbro, altrimenti viene usato `XTTS_SPEAKER` (predefinito `Dionisio Schuyler`). Esporta queste variabili nell'ambiente del processo TTS: un servizio avviato separatamente non eredita automaticamente il caricamento di `backend/.env` eseguito dal backend Node. Prima della sintesi vengono scaricati da Ollama solo i modelli attivi indicati da `OLLAMA_VISION_MODEL` e `OLLAMA_TEXT_MODEL`; imposta `TTS_UNLOAD_OLLAMA_MODELS=0` per disattivare questa operazione. XTTS viene caricato su richiesta e rilasciato dopo ogni sintesi (`XTTS_UNLOAD_AFTER_SYNTHESIS=1`); Whisper viene spostato temporaneamente su CPU e ripristinato dopo la generazione, se la VRAM lo consente. Se CUDA, XTTS o la VRAM libera non sono disponibili, viene usato Piper su CPU (`PIPER_LENGTH_SCALE=1.28`), che supporta solo l'italiano. `TTS_AUDIO_FILTER` consente filtri FFmpeg opzionali; per impostazione predefinita non viene applicato alcun pitch shift o compressione audio. Whisper trascrive la narrazione in italiano con timestamp parola per parola; i sottotitoli usano le parole esatte dello script, gruppi fino a tre parole, un anticipo di circa 50 ms e intervalli senza sovrapposizioni. La narrazione punta a 50 secondi, entro 40-60 secondi, e chiede a `llama3.1` una storia horror originale di 90-125 parole, con inizio, tensione crescente e finale d'effetto.

Prima della sintesi, il testo narrativo viene ripulito da Markdown e simboli non pronunciabili, riceve pause di respiro nelle frasi troppo lunghe e una punteggiatura finale esplicita; lo stesso testo pulito viene usato per TTS e sottotitoli. Il face tracking tiene conto della larghezza del volto per mantenere un margine nel crop, conserva il punto di fuoco per 1,5 secondi di rilevamenti mancanti e poi torna gradualmente al punto focale della scena. Il render applica scanline e grana analogica all'intera sequenza prima di sovrapporre i sottotitoli; puoi regolarle con `HORROR_OVERLAY_ENABLED=1`, `HORROR_OVERLAY_MODE=analog|noise|scanlines`, `HORROR_OVERLAY_NOISE=4` e `HORROR_OVERLAY_SCANLINE_OPACITY=0.12`.

XTTS usa `XTTS_TEMPERATURE=0.65`, `XTTS_TOP_K=50`, `XTTS_TOP_P=0.85`, `XTTS_REPETITION_PENALTY=2.0` e `XTTS_SPEED=0.92` come valori predefiniti. Per correggere pronunce specifiche, `XTTS_ITALIAN_PRONUNCIATION_OVERRIDES` accetta una mappa JSON parola/sostituzione. Per `VOICE_REFERENCE_WAV` vengono verificati formato WAV, durata (1-30 secondi), frequenza, livelli e clipping; il servizio registra durata e sample rate. Musica e riverbero marcato richiedono comunque un ascolto manuale del campione.

Per XTTS italiano, le ore, i numeri, le percentuali, le abbreviazioni e alcune sigle vengono espansi prima della sintesi. Il testo è diviso in segmenti di massimo 28 parole rispettando frasi e pause; brevi silenzi (150-280 ms) tra segmenti rafforzano la punteggiatura.

<!-- Le risposte Qwen vengono lette da `message.content` dell'API `/api/chat`; il parser accetta anche JSON recintato in Markdown, array di momenti e nomi alternativi per descrizioni e fotogrammi. I campi mancanti ricevono fallback conservativi (`has_text=true`, `is_animated=false`); risposte non conformi producono un avviso con un breve estratto e continuano se resta una descrizione utilizzabile. Solo una risposta vuota o priva di contenuto visivo utilizzabile blocca l'analisi. -->

Per `/api/chat`, Qwen riceve uno schema JSON che richiede `scene_summary`, `actions`, `moments` e i relativi campi. Se il modello non lo rispetta, il parser accetta fence Markdown, array e alias; registra l'estratto grezzo della risposta. I fallback inferiscono “nessun testo” o “animato” solo quando la descrizione lo afferma chiaramente; in assenza di prove restano conservativi. Solo una risposta vuota o priva di contenuto visivo utilizzabile blocca l'analisi.

## Download e licenze

Per i video che richiedono l'accesso, usa **Login YouTube (Cache)** nella UI: puoi importare la
sessione da Chrome, Edge o Firefox, oppure caricare un `cookies.txt` in formato Netscape. La cache
locale predefinita è `data/youtube_cookies.txt` (ignorata da Git); per cambiarla imposta
`YOUTUBE_COOKIES_FILE`. La UI usa `POST /api/youtube/auth` per importare e
`GET /api/youtube/auth/status` per verificare la cache. I cookie sono conservati solo dal backend e
non vengono restituiti dall'API. Se il browser blocca il database, chiudilo completamente o usa
l'importazione del file. La UI locale su `http://localhost:4200` è autorizzata per l'import; per
un'origine diversa, configura `YOUTUBE_AUTH_ALLOWED_ORIGINS` con le origini frontend consentite,
separate da virgole.

La ricerca seleziona esclusivamente video il cui campo licenza di YouTube dichiara una licenza Creative Commons Attribution; risultati senza licenza verificabile vengono ignorati. I file vengono salvati in `downloads/` (massimo 3 per richiesta). Verifica sempre i termini applicabili, i diritti e gli obblighi di attribuzione prima di usare o ridistribuire i contenuti. Questo starter non aggira restrizioni di accesso o protezioni tecniche.

Il downloader prova formati MP4/M4A e ripiega su flussi video/audio WebM o altri formati disponibili. yt-dlp usa FFmpeg per unire e rimuxare i risultati in MP4.

La ricerca yt-dlp usa `extract_flat` per evitare l'estrazione completa di ogni risultato e limita i candidati a 5. Per ridurre i rate-limit, ricerca e download impostano `sleep_interval_requests=1.5`, `sleep_interval=3` e `max_sleep_interval=6`.
