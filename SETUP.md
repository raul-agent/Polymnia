# POLYMNIA — entorno local (uv, CLI)

Montaje del pipeline en esta máquina con **uv** y acceso 100 % por línea de comandos.
Todo se invoca con `./bin/polymnia <comando>`; no hay que activar ningún venv.

```bash
cd Polymnia
./bin/polymnia doctor      # qué hay instalado y qué falta
./bin/polymnia help
```

## Qué se instaló

| Componente | Cómo | Dónde |
|---|---|---|
| Python 3.12.14 + deps | `uv sync` (`pyproject.toml` → `uv.lock`) | `.venv/` |
| PyTorch 2.8.0 **cu128** | índice `pytorch-cu128` en `pyproject.toml` | `.venv/` (CUDA OK, RTX 4090) |
| WhisperX | `git+https://github.com/m-bain/whisperx.git` | consola `whisperx` en el venv |
| R: `multimolang`, `logger` | `install.packages(lib="R_libs")` | `R_libs/` (`arrow` ya estaba en el sistema) |
| `config_dfMaker.json` | copia de `R_libs/multimolang/extdata/config_all_true.json` | raíz del repo |
| parole (prosodia) | `git clone daedalusLAB/parole` | `create_datasets/parole/` |
| Praat 6.4.30 barren (sin root) | tarball de GitHub releases | `create_datasets/parole/env/praat/praat` |
| DuckDB CLI v1.5.6 | binario oficial (el del sistema es un snap que no arranca) | `tools/bin/duckdb` |
| OpenPose | **ya existía** en el sistema | `/opt/openpose/build/examples/openpose/openpose.bin` |
| TalkNet-ASD | `git clone` del fork de RaulKite, repo **hermano** | `../TalkNet-ASD` (`POLYMNIA_ASD_PATH`) |
| Pesos ASD (`sfd_face.pth`, `pretrain_TalkSet.model`) | `gdown` desde Drive, **ya descargados** | dentro de `../TalkNet-ASD` |

Ficheros añadidos por este montaje (no tocan el código original del pipeline):
`pyproject.toml`, `uv.lock`, `.python-version`, `config_dfMaker.json`,
`bin/polymnia`, `tools/polymnia_prosody_compat.py`, `tools/check_tab.py`,
`tools/verify_videos.py`, `tools/bin/duckdb`.

TalkNet-ASD vive **fuera** de este repo y reutiliza su venv: `uv sync --extra asd`
(añade `scenedetect`, `python_speech_features`, `facenet-pytorch`, `gdown`).

## Comandos

```
polymnia doctor                         estado del entorno
polymnia check-tab F [--offset N] [--show N]   # validar tabulado sin descargar
polymnia download  --txt_file F --searchterm T --output_dir D [--offset N] [--no-check]
polymnia asd       --input_dir D --output_dir D [--second N]   # TalkNet-ASD: raw -> masked
polymnia filter    --videos_folder D --matched_videos D --discarded_videos D [--check_hands True]
polymnia argos     --videos_folder D --output_folder D [--openpose_path P] [--face_hands True] [--skeletons True]
polymnia parquet   --json_folder D --dataset_dir D       # solo el paso R (dfMaker)
polymnia ecos      --input_folder D --n_persons 1        # prosodia: Silero + Praat + R
polymnia apate     --input_folder D --n_persons 1        # WhisperX + palabras por frame
polymnia voice     --input_folder D --n_persons 1        # ecos + apate en paralelo
polymnia merge     --input_folder D --n_persons 1 --db_name D.duckdb [--no-compat]
polymnia whisperx  <args...>                             # whisperx directo (cuda + fp16)
polymnia duckdb    <ruta.duckdb> [-c "SELECT ..."]
polymnia rscript   <script.R> [args...]                  # Rscript con R_libs del repo
polymnia python    <args...>                             # python del venv
```

El wrapper pone las variables que los scripts del repo dan por sentadas:
`R_LIBS_USER=R_libs`, `PYTHON_SCRIPT_DIR=<repo>` (el R la usa para encontrar
`R_libs`), `PATH` con `praat`, `whisperx` y el duckdb local.

## Flujo completo

```bash
OUT=/ruta/a/out
./bin/polymnia check-tab  clips.txt                      # antes de descargar nada
./bin/polymnia download   --txt_file clips.txt --searchterm clima --output_dir $OUT/videos/raw
./bin/polymnia python     tools/verify_videos.py $OUT/videos/raw   # mp4 reales vs HTML de error
./bin/polymnia asd        --input_dir $OUT/videos/raw --output_dir $OUT/videos/masked
./bin/polymnia filter     --videos_folder $OUT/videos/masked \
                          --matched_videos $OUT/videos/1_person \
                          --discarded_videos $OUT/videos/discarded
./bin/polymnia argos      --videos_folder $OUT/videos/1_person --output_folder $OUT
./bin/polymnia voice      --input_folder $OUT --n_persons 1
./bin/polymnia merge      --input_folder $OUT --n_persons 1 --db_name $OUT/dataset.duckdb
./bin/polymnia duckdb $OUT/dataset.duckdb -c "SELECT count(*) FROM multi_data"
```

`create_dataset.py --input clips.txt --output_folder $OUT --searchterm clima` también
funciona ya con ASD instalado: `ASD_PATH` se resuelve con `POLYMNIA_ASD_PATH`
(antes apuntaba a `/home/user/TalkNet-ASD`).

El `--n_persons` de `voice`/`merge` tiene que coincidir con lo que **OpenPose**
detectó, no con lo que filtraste: `argos` clasifica en `{n}_persons/` y puede
acabar en `2_persons` aunque `filter` viera una sola persona (la caja expandida
del enmascarado puede arrastrar parte de un segundo cuerpo).

**Una pasada de `voice`/`merge` solo cubre una categoría.** Tras `argos` hay que
repetirlo por cada `{n}_persons` que exista, cada uno con su propia base de datos:

```bash
for n in 1 2 3 4; do
  [ -d $OUT/dataset/${n}_persons/parquet_files ] || continue
  ./bin/polymnia voice --input_folder $OUT --n_persons $n
  ./bin/polymnia merge --input_folder $OUT --n_persons $n --db_name $OUT/multi_${n}p.duckdb
done
```

No se debe mezclar todo en una sola tabla: cada clip lleva repetidas sus filas por
persona y por punto articular, así que el número de filas por clip depende de cuánta
gente saliera en pantalla. En la prueba: `1p` 819.397 filas/10 clips, `2p` 455.114/5,
`3p` 410.589/2, `4p` 158.509/1.

## Cinco bugs reales encontrados (y qué se hizo)

1. **`download_clips.py` guardaba HTML como `.mp4`.** El CGI de NewsScape responde
   **HTTP 500 con un cuerpo HTML**, y `curl` sin `--fail` sale con 0: el script daba
   la descarga por buena, `os.path.getsize() > 0` se cumplía y el **bucle de reintentos
   no llegaba a activarse**. Resultado: carpetas llenas de `.mp4` que son HTML y que
   OpenPose/WhisperX fallan más tarde en silencio. Corregido en el repo: `--fail`,
   comprobación de tamaño y borrado del fichero parcial. `tools/verify_videos.py`
   audita un directorio ya descargado (`OK/EMPTY/NOTMP4/NOVIDEO/BAD`, `--delete-bad`).

2. **TalkNet-ASD elegía mal al hablante.** `process_video.py` asumía que
   `tracks[0]` era el que habla, pero TalkNet emite los tracks en **orden
   arbitrario**. En el clip de demo del propio repo (`demo/001.avi`) el hablante es
   el track 1 (score **+1.79**) y el código miraba el 0 (**-2.12**), así que
   **descartaba clips válidos** sin avisar. Se añadió `pick_active_track()`: elige el
   track con mejor score entre los visibles en ese frame. Verificado: el clip pasa de
   descartado a procesarse (`Active track: 1`, score **+1.57**), y el enmascarado deja
   solo al hablante (1 % → 47 % de píxeles negros).

3. **`os.rename` cruzaba sistemas de ficheros.** `./output/x.mp4 -> /data/.../masked/`
   reventaba con `[Errno 18] Invalid cross-device link` justo al final, con el vídeo
   ya generado. Cambiado a `shutil.move`.

4. **El enmascarado re-temporizaba el clip.** `create_tracked_person_video` hacía
   `fps = int(cap.get(cv2.CAP_PROP_FPS))`: con un NewsScape a 29.97 escribía a **29 fps**,
   así que los mismos 663 frames duraban 22.86 s en vez de 22.12 s. El número de frame
   no se altera (el join por `frame` de `hefesto` sigue siendo correcto, verificado en
   los 26 clips), pero la duración que declaraba el `.mp4` era mentira para cualquiera
   que la leyera. Ahora se pasa el fps en float. Verificado: raw y masked quedan a
   `2997/100` y 22.122 s idénticos.

5. **`boxes.id` puede ser `None`.** En frames donde el tracker de YOLO no tiene pista
   confirmada, `frame_results.boxes.id` es `None` y `zip(boxes, ids)` reventaba con
   `'NoneType' object is not iterable`, **perdiendo el clip entero** tras haber hecho
   todo el trabajo previo. Se trata como frame negro (y `get_track_id_at_point`
   devuelve `None`).

Además en el fork de TalkNet, necesario para arrancar aquí: `scenedetect` ≥0.7 exige
`fps` float (`scene_utils.py`), `show=True` en YOLO aborta el proceso en headless
(`cv2.imshow` sin plataforma Qt) y `run_asd_for_custom_video_in_local.sh` tenía la ruta
absoluta del autor (`cd /data/home/raul/...`) y un `/tmp/talknet_processing` compartido
que se pisaba entre ejecuciones concurrentes.

## Dos desajustes del repo de Polymnia (esquivados, no modificados)

1. **`ecos.py` ↔ `hefesto.py`.** `parole.sh` escribe la prosodia en
   `parquet_speech_analysis/{id}/{id}_prosody.parquet` con columna `frame_id`;
   `hefesto.py` busca `parquet_speech_analysis/{id}.parquet` y renombra `Frame`.
   Un run completo terminaba siempre con `[WARNING] Prosody file not found` y el
   DuckDB salía vacío. `merge` ejecuta antes
   `tools/polymnia_prosody_compat.py`, que genera el layout plano derivado del
   original (idempotente, con manifiesto, no borra lo que no generó él).
   `--no-compat` lo desactiva. Se probó end-to-end: `Total discrepancies: 0`.

2. **`config_dfMaker.json` no está en el repo** pero `max_people_classification.R`
   lo exige en el directorio de trabajo. Se creó en la raíz a partir de
   `config_all_true.json` que trae el paquete `multimolang`
   (extrae datetime, país, cadena, programa, hora…). Revísalo si tus nombres de
   fichero no siguen el patrón UCLA NewsScape: desactiva las claves que no apliquen.

## Muestreo de prosodia: por qué `hefesto.py` ya no hace `iloc[::6]`

`prepare_prosody_data` reducía la prosodia **posicionalmente** (`iloc[::6]`), es
decir «una fila de cada seis». Eso solo equivale a «un muestreo por frame de
vídeo» si el productor emite exactamente 6 filas por frame y además llegan ordenadas.
Con el pipeline actual no se cumple: `process_prosody.R` interpola (spline) pitch,
intensity y harmonicity **sobre los timestamps de frame** antes de escribir, o sea
1 fila por frame (verificado: 485 filas para 485 frames consecutivos). El `::6`
sobre eso era un **doble recorte**: tiraba 5/6 de mediciones válidas y dejaba `pitch`
en solo el **16,5 %** de las filas unidas.

Ahora se selecciona **por el valor de `frame`** (`drop_duplicates` + `sort`), que es
identidad cuando los datos ya vienen a 1 fila/frame y sigue dando una fila por frame
si volviera un productor denso. Ateado en `tools/test_prosody_align.py` (3 casos:
1 fila/frame, denso 6/frame, denso barajado — los dos ultimos fallaban con `::6`).

Medido sobre los 10 clips reales de `1_persons`, mismo join y mismas filas:

| | filas | filas con `pitch` | tamano |
|---|---|---|---|
| con `iloc[::6]` | 819.397 | 135.767 (16,6 %) | 24,5 MB |
| seleccionando por `frame` | 819.397 | 818.027 (99,8 %) | 24,8 MB |

**Lee la columna con cuidado: 99,8 % de filas *con valor* no es 99,8 % de medición
real.** El spline de parole rellena el silencio, así que el **45,8 %** de los `pitch`
lleva un valor interpolado cerca de cero (`pitch < 20`), y solo el **53,9 %** es una
medición plausible. **Filtra por `vad` antes de entrenar nada fonético**: con
`vad = false` el pitch medio sale 7,8 Hz (no es fonación). Eso es comportamiento de
parole, anterior a este cambio, y no se toca aquí.

## La última etapa: anotación visual con VLM (`theia`)

Después de `merge`, `theia.py` anota cada vídeo que llegó al final preguntando a
un LLM con visión compatible con la API de OpenAI (vLLM). Dos llamadas por vídeo,
cada una con las imágenes que le sirven: la **escena** ve los 5 frames del vídeo
raw (`indoor_outdoor`, `show_type`); el **hablante** ve los 5 del masked + los 5
del raw como contexto (`hands_free`, `sitting_standing`, `screen_interaction`,
`sex`, `age`). Los 5 frames son primero, último y 3 intermedios proporcionales.

La respuesta se valida contra un esquema estricto (enums exactos, edad entera
0-120, booleanos reales, ni claves de más ni de menos); si no cumple, se reintenta
(`--attempts`, defecto 3) **diciéndole al modelo por qué falló la anterior**. Sin
validación no entra nada: el criterio de aceptación es sintaxis, no calidad.

Salidas: CSV `dataset/{n}_persons/video_annotations.csv` (una fila por id con
`status` ok|error; sirve de resume: los ok se saltan al relanzar) y DuckDB:
`multi_data` gana las 7 columnas (consultables junto a todo lo demás) y se crea
la tabla `video_annotations` (una fila por vídeo).

```
cp .env.example .env    # y rellenar THEIA_API_BASE / THEIA_MODEL / THEIA_API_KEY
./bin/polymnia theia --input_folder D --n_persons 1 --db_name D/dataset.duckdb
```

El `.env` está excluido de git (verificado con `git check-ignore`); la plantilla
subible es `.env.example`. `--api-base/--api-key/--model` pisan al entorno si
prefieres no usar fichero.

Ateado en `tools/test_theia.py`: validador (coerciones, nullish, 7 rechazos),
muestreo de frames, limpieza de respuestas con cercas/texto alrededor, y un e2e
contra un servidor OpenAI-compatible falso que primero responde basura para
probar el reintento, y después comprueba CSV, columnas en `multi_data`, tabla,
y que el resume no vuelve a llamar. `polymnia test`.

## El divisor de 1000/100 es correcto: no lo "arregles"

`download_clips.py` lee los campos 4º/5º según el **nombre** del fichero (columna 1):

```python
divisor = 1000 if name.startswith("2017") else 100
```

Esto es intencionado y correcto, no un bug:

* **2017 en adelante** → el campo se **llama** `centiseconds` por compatibilidad
  retroactiva, pero **contiene milisegundos de verdad**. Por eso se divide entre 1000.
  Se ve en el propio nombre de clip: `..._356.505_361.531_before` = seg.ms,
  y con `--offset 2` salen `354.505`/`363.531`. ✔
* **Antes de 2017** → son centésimas reales, se divide entre 100.

La consecuencia práctica es que **el ancho de columna es válido en su convención**:
`505` en un fichero 2017 son 0.505 s, y `50` en uno antiguo son 0.50 s. Lo que **sí**
rompe una línea es mezclar convenciones o leer el campo con el divisor equivocado:
entonces `start > end` y `newsscape_mp4_snippet.cgi` responde con un **HTTP 500 y
cuerpo HTML** (que ya no se guarda gracias a la corrección del bug 1).

`check-tab` comprueba el divisor y, además, contrasta los números con el rango que el
propio nombre del clip lleva dentro (`..._356.505_361.531_...`), que es una fuente
independiente de verdad. No toca la red:

```bash
./bin/polymnia check-tab /ruta/tu_tabulado.txt --offset 2 --show 5
# filas=120 mal_formadas=0 invertidas=0 sin_coincidir_con_nombre=0
```

`download` lo ejecuta automáticamente antes de descargar (se salta con `--no-check`).
Si aparecen `MAL` en ficheros 2017, sospecha de un tabulado que llegó con centésimas
(aunque la columna se llame igual) antes de tocar el código: normaliza el fichero o
ajusta el divisor, y comprueba el resultado contra el rango del nombre.

## Lo que falta / no está probado

* **`process_video.py` sigue teniendo criterios propios de descarte**: exige escena
  única y score > 0 en el frame de `--second` (por defecto el 2). Un clip cuyo
  hablante hable más tarde puede descartarse; sube `--second` si ves falsos negativos.
* **ASD en clips reales de NewsScape** no se ha probado todavía (solo con el
  `demo/001.avi` del repo y recortes suyos), porque las descargas dependenden de que
  el CGI responda; se probará con tus tabulados.
* **Modelo WhisperX `large` + alineación**: la transcripción ya se validó en GPU
  (`faster-whisper-large-v3` en caché). La primera vez que `apate.py` pida el modelo
  de alineación para un idioma nuevo hará falta red y unos minutos.
* **R 4.1.2** es viejo; `multimolang`/`arrow` funcionan, pero si algo pide R ≥ 4.3
  habrá que actualizar R (necesita root).
* El `duckdb` del sistema es un snap que falla en este sandbox (`mount --rbind`);
  se usa `tools/bin/duckdb`.

## Verificación realizada

Pipeline completo ejecutado con un clip sintético (60 frames @25 fps, audio real
de 2.4 s) + keypoints OpenPose de ejemplo de `multimolang`:

`dfMaker/R → parquet` → `ecos (Silero+Praat+R)` → `apate (WhisperX GPU)` → `merge`
→ `multi_data` con **8220 filas**, pose + `pitch`/`intensity` + `words` alineados
por `frame`, 0 discrepancias.

**Segunda pasada, con TalkNet-ASD incluido** (clip real con hablante, 100 frames
@25 fps, a partir del `demo/001.avi` del propio TalkNet):

`asd → filter → argos (OpenPose GPU) → ecos → apate → merge`
→ `multi_data` con **14248 filas**, pose + `pitch`/`intensity` + `words` + `datetime`
alineados por `frame`, **0 discrepancias**. ASD seleccionó el hablante correctamente
(`Active track: 0`, score +1.05) y el enmascarado recorta al hablante.

Además: YOLO (`filter`) clasifica correctamente un vídeo con 5 personas a
`discarded/5` y acepta uno de una (`SHOULDERS AND HANDS: True`); OpenPose corre y
detecta GPU; WhisperX CLI transcribe y alinea; `verify_videos.py` distingue vídeo
bueno de HTML de error, vacío y truncado; la descarga con clip id inválido ya no deja
ningún fichero basura en disco (3 reintentos y nada).

Rendimiento observado en esta máquina: OpenPose 100 frames ≈ 9 s; TalkNet ≈ 6 s de
inferencia sobre 502 frames; WhisperX `large` en GPU ≈ 10 s por clip corto.
