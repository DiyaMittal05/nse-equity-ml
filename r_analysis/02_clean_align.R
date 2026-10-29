#!/usr/bin/env Rscript
# 02_clean_align.R — Clean, align, compute log-returns, summary stats

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

MIN_ROWS  <- 120
if (!dir.exists(CLEAN_DIR)) dir.create(CLEAN_DIR, recursive = TRUE)

csv_files <- list.files(RAW_DIR, pattern = "\\.csv$", full.names = TRUE)
csv_files <- csv_files[!grepl("_fetch_log", csv_files)]

cat("Raw files found:", length(csv_files), "\n\n")

close_list <- list()
ohlc_list  <- list()
meta_log   <- data.frame(ticker = character(), rows_raw = integer(), rows_clean = integer(),
                         start_date = as.Date(character()), end_date = as.Date(character()),
                         status = character(), stringsAsFactors = FALSE)

for (f in csv_files) {
  ticker <- gsub("\\.csv$", "", basename(f))
  cat("Processing:", ticker, "... ")
  df <- tryCatch(read.csv(f, row.names = 1, stringsAsFactors = FALSE), error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) {
    cat("EMPTY\n")
    meta_log <- rbind(meta_log, data.frame(ticker = ticker, rows_raw = 0, rows_clean = 0,
      start_date = NA, end_date = NA, status = "EMPTY", stringsAsFactors = FALSE))
    next
  }
  dates <- as.Date(rownames(df))
  if (any(is.na(dates))) { dates <- as.Date(df[[1]]); df <- df[, -1, drop = FALSE] }
  close_col <- grep("Close$", colnames(df), value = TRUE)
  if (length(close_col) == 0) {
    cat("NO Close COLUMN\n")
    meta_log <- rbind(meta_log, data.frame(ticker = ticker, rows_raw = nrow(df), rows_clean = 0,
      start_date = NA, end_date = NA, status = "NO_CLOSE", stringsAsFactors = FALSE))
    next
  }
  close_vec <- as.numeric(df[[close_col]])
  xts_close <- xts(close_vec, order.by = dates); colnames(xts_close) <- ticker
  xts_ohlc  <- xts(df, order.by = dates)
  if (nrow(xts_close) < MIN_ROWS) {
    cat("SHORT HISTORY (", nrow(xts_close), "rows)\n")
    meta_log <- rbind(meta_log, data.frame(ticker = ticker, rows_raw = nrow(df),
      rows_clean = nrow(xts_close), start_date = min(dates), end_date = max(dates),
      status = "SHORT", stringsAsFactors = FALSE))
    next
  }
  close_list[[ticker]] <- xts_close
  ohlc_list[[ticker]]  <- xts_ohlc
  cat("OK (", nrow(xts_close), "rows)\n")
  meta_log <- rbind(meta_log, data.frame(ticker = ticker, rows_raw = nrow(df),
    rows_clean = nrow(xts_close), start_date = min(dates), end_date = max(dates),
    status = "OK", stringsAsFactors = FALSE))
}

cat("\nAligning to common date index...\n")
all_dates <- sort(unique(do.call(c, lapply(close_list, index))))
close_aligned <- xts(order.by = all_dates)
for (ticker in names(close_list)) {
  aligned <- merge(close_list[[ticker]], xts(order.by = all_dates), join = "right")
  colnames(aligned) <- ticker
  close_aligned <- merge(close_aligned, aligned)
}
if (ncol(close_aligned) > length(close_list)) close_aligned <- close_aligned[, -1]
cat("Aligned panel:", nrow(close_aligned), "dates x", ncol(close_aligned), "tickers\n")

cat("Computing daily log-returns...\n")
returns <- log(close_aligned / lag(close_aligned, k = 1))
returns <- returns[-1, ]

cat("Building summary statistics...\n")
ann_ret <- colMeans(returns, na.rm = TRUE) * 252 * 100
ann_vol <- apply(returns, 2, sd, na.rm = TRUE) * sqrt(252) * 100
sharpe  <- ann_ret / ann_vol
max_dd  <- sapply(close_aligned, function(x) {
  x <- na.omit(x)  # newer listings have leading NAs, which would make cummax all NA
  cummax_x <- cummax(x)
  dd <- (x - cummax_x) / cummax_x
  min(dd, na.rm = TRUE) * 100
})

summary_df <- data.frame(ticker = names(ann_ret), rows = colSums(!is.na(returns)),
  ann_return_pct = round(ann_ret, 2), ann_volatility_pct = round(ann_vol, 2),
  sharpe_ratio = round(sharpe, 2), max_drawdown_pct = round(max_dd, 2),
  stringsAsFactors = FALSE)
summary_df <- summary_df[order(-summary_df$ann_return_pct), ]

write.csv(as.data.frame(close_aligned), file.path(CLEAN_DIR, "close_prices_aligned.csv"))
write.csv(as.data.frame(returns),      file.path(CLEAN_DIR, "daily_log_returns.csv"))
write.csv(meta_log,                     file.path(CLEAN_DIR, "_cleaning_log.csv"), row.names = FALSE)
write.csv(summary_df,                   file.path(CLEAN_DIR, "summary_stats.csv"),  row.names = FALSE)
for (ticker in names(ohlc_list)) {
  write.csv(as.data.frame(ohlc_list[[ticker]]), file.path(CLEAN_DIR, paste0(ticker, "_ohlcv.csv")))
}

cat("\nStep 1 complete.\nFiles saved to:", CLEAN_DIR, "\n")
cat("Top 5 performers by annualized return:\n")
print(head(summary_df, 5))
