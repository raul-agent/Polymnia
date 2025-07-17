# Obtener el directorio donde está este script de Python
python_script_dir <- normalizePath(Sys.getenv("PYTHON_SCRIPT_DIR"), mustWork = FALSE)

# Definir un path relativo dentro de create_datasets
user_lib_path <- file.path(python_script_dir, "R_libs")

# Crear el directorio si no existe
if (!dir.exists(user_lib_path)) {
  dir.create(user_lib_path, recursive = TRUE)
}

# Asegurar que R utilice esta librería
.libPaths(user_lib_path)

check_and_install <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package, lib = user_lib_path, dependencies = TRUE)
  }
  library(package, character.only = TRUE)
}

# Uso normal de la función
check_and_install("multimolang")
check_and_install("logger")
check_and_install("arrow") 


# Obtener argumentos de línea de comandos
args <- commandArgs(trailingOnly = TRUE)
input_folder <- args[1]
expression_year_dir <- args[2]

log_info("Input folder: {input_folder}")
log_info("Expression year directory: {expression_year_dir}")

# Ya no se carga dfMaker desde un .rda, sino desde multimolang directamente
log_info("multimolang package loaded successfully")

# Process the directory content with dfMaker
tryCatch({
  log_info("Starting dfMaker processing")
  log_info("Contents of input folder: {paste(list.files(input_folder), collapse=', ')}")
  
  clip <- dfMaker(
    input.folder = input_folder,
    no_save = TRUE,
    config.path = "config_dfMaker.json"
  )
  
  log_info("dfMaker processing completed")
  log_info("Structure of clip: {capture.output(str(clip))}")
  
}, error = function(e) {
  log_error("Error in dfMaker: {conditionMessage(e)}")
  log_error("Traceback: {paste(capture.output(traceback()), collapse='\n')}")
  stop("Execution interrupted due to an error in dfMaker")
})

# Obtener el id del clip
clip$id

# Número total de personas
num_people <- max(clip$people_id)

# Definir la ruta de salida usando expression_year_dir
output_path <- file.path(
  expression_year_dir,
  paste0(num_people, "_persons"),
  "parquet_files",
  paste0(unique(clip$id), ".parquet")
)

# Crear el directorio si no existe
output_dir <- dirname(output_path)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Escribir el dataframe en un archivo Parquet
write_parquet(x = clip, sink = output_path)

# Guardar el número de personas en un archivo para uso posterior
num_people_file <- file.path(input_folder, "num_persons.txt")
write(num_people, file = num_people_file)

cat("Archivo procesado y guardado en:", output_path, "\n")
