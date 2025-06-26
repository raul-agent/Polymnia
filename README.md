# Cosas que faltan

Paarole no está, pero se descarga y se añade sin más.
Tengo que revisar en ecos.py que llame bien a Parole
Instalador automático de TalkNet--ASD
Direcotrio "dependences" que contenga los paquetes de Python (whisperX dentro de Python), R, TalkNet--ASD, Parole, OpenPose y PRAAT

dependences/
└── env/
    ├── pyenv/                         # entorno virtual de Python
    │   ├── bin/
    │   ├── lib/
    │   │   └── python3.10/
    │   │       └── site-packages/
    │   │           └── whisperx/      # paquete instalado en editable mode
    │   └── requirements.txt           # especifica versiones exactas
    │
    ├── Rlibs/                         # bibliotecas locales de R
    │   ├── renv.lock                  # snapshot reproducible (renv)
    │   └── <paquetes_compilados>
    │
    ├── talknet-asd/                   # entorno aislado de TalkNet-ASD
    │   ├── checkpoints/               # modelos preentrenados
    │   ├── config/                    # YAML/JSON para inferencia
    │   └── scripts/                   # wrappers de entrenamiento
    │
    └── openpose/                      # binarios y recursos de OpenPose
        ├── bin/                       # ejecutables (CPU/GPU)
        ├── models/                    # pesos de redes neuronales
        └── examples/                  # ejemplos de línea de comandos
