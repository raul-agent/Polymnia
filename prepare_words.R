prepare_words_data <- function(words_path) {
  words <- read_json(words_path)
  processed_words <- do.call(rbind, lapply(words, function(frame) {
    words <- paste(frame$words, collapse = ", ")
    scores <- paste(frame$scores, collapse = ", ")
    data.frame(
      frame = frame$frame_number,
      words = words,
      scores_w = scores,  # Renaming scores to scores_w
      stringsAsFactors = FALSE
    )
  }))
  return(processed_words)
}
