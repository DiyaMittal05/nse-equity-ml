#!/usr/bin/env Rscript
# 01_fetch_data.R — Fetch daily OHLCV for a universe (psu | nbfc) from Yahoo Finance

pkgs <- c("quantmod", "xts", "zoo")
for (p in pkgs) {
  if (!require(p, character.only = TRUE, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org/", quiet = TRUE)
    library(p, character.only = TRUE)
  }
}

# Locate this script so paths work from any working directory
.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
SCRIPT_DIR <- if (length(.file_arg)) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else file.path(getwd(), "r_analysis")
source(file.path(SCRIPT_DIR, "config.R"))

START_DATE <- as.Date("2019-01-01")
END_DATE   <- Sys.Date()
if (!dir.exists(RAW_DIR)) dir.create(RAW_DIR, recursive = TRUE)

# TICKERS comes from config.R

cat("Tickers to fetch:", length(TICKERS), "\n")

master_log <- data.frame(ticker = character(), rows_fetched = integer(),
                         start_date = as.Date(character()), end_date = as.Date(character()),
                         status = character(), stringsAsFactors = FALSE)

for (sym in TICKERS) {
  cat("Fetching:", sym, "... ")
  result <- tryCatch({
    data <- getSymbols(Symbols = sym, src = "yahoo", from = START_DATE, to = END_DATE,
                       auto.assign = FALSE, verbose = FALSE)
    if (is.null(data) || nrow(data) == 0) stop("Empty response")
    colnames(data) <- gsub(paste0(sym, "."), "", colnames(data), fixed = TRUE)
    # Drop bars with no close (Yahoo returns the latest, not-yet-final session that way)
    data <- data[!is.na(data[, "Close"]), ]
    out_path <- file.path(RAW_DIR, paste0(sym, ".csv"))
    write.csv(as.data.frame(data), out_path, row.names = TRUE)
    cat("OK (", nrow(data), "rows)\n")
    data.frame(ticker = sym, rows_fetched = nrow(data),
               start_date = as.Date(index(data)[1]), end_date = as.Date(index(data)[nrow(data)]),
               status = "OK", stringsAsFactors = FALSE)
  }, error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    data.frame(ticker = sym, rows_fetched = 0, start_date = NA, end_date = NA,
               status = paste0("ERROR: ", conditionMessage(e)), stringsAsFactors = FALSE)
  })
  master_log <- rbind(master_log, result)
  Sys.sleep(0.5)
}

log_path <- file.path(RAW_DIR, "_fetch_log.csv")
write.csv(master_log, log_path, row.names = FALSE)
cat("\nFetch complete. Successful:", sum(master_log$status == "OK"),
    "Failed:", sum(master_log$status != "OK"), "\n")
