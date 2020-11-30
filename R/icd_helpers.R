library(icd)

get_possible_codes <- function(include_patterns, exclude_patterns) {
  tables <- list(icd9cm_hierarchy, icd10cm2019)
  columns <- c("long_desc")
  results <- list()

  i <- 1
  for (table_ in tables) {
    row_is_match <- rep(FALSE, nrow(table_))
    for (col_ in columns) {
        all_patterns_match <- rep(TRUE, nrow(table_))

        for(pattern in include_patterns) {
            p_matches <- grepl(pattern, table_[, col_], ignore.case=TRUE, perl=TRUE)
            all_patterns_match <- all_patterns_match & p_matches
        }

        for(pattern in exclude_patterns) {
            p_matches <- !grepl(pattern, table_[, col_], ignore.case=TRUE, perl=TRUE)
            all_patterns_match <- all_patterns_match & p_matches
        }


        row_is_match <- row_is_match | all_patterns_match
    }
      matching_rows <- table_[row_is_match, c("code", "short_desc", "long_desc")]
      results[[i]] <- matching_rows
      i <- i + 1
  }

    do.call(rbind, results)
}


get_codes_by_pattern <- function(pattern) {
    tables <- list(icd9cm_hierarchy, icd10cm2019)
    results <- list()
    i <- 1
    for (table_ in tables) {
      is_match <- grepl(pattern, table_[, "code"], ignore.case=TRUE, perl=TRUE)
      matching_rows <- table_[is_match, c("code", "long_desc")]
      results[[i]] <- matching_rows
      i <- i + 1
     }
    do.call(rbind, results)
}
