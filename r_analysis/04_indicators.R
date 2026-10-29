#!/usr/bin/env Rscript
# 04_indicators.R — Full technical indicator suite for every ticker in the universe
# SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, Stochastic, OBV

pkgs <- c("quantmod", "TTR", "xts", "zoo")
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

clean_dir <- CLEAN_DIR
plot_dir  <- PLOT_DIR
ind_dir   <- IND_DIR
for (d in c(plot_dir, ind_dir)) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

ohlcv_files <- list.files(clean_dir, pattern = "_ohlcv\\.csv$", full.names = FALSE)
tickers <- gsub("_ohlcv\\.csv$", "", ohlcv_files)
cat("Tickers found:", length(tickers), "\n\n")

read_ohlcv <- function(ticker) {
  f <- file.path(clean_dir, paste0(ticker, "_ohlcv.csv"))
  if (!file.exists(f)) return(NULL)
  df <- read.csv(f, stringsAsFactors = FALSE)
  dates <- as.Date(df[[1]]); df <- df[, -1, drop = FALSE]
  names(df) <- sub("^[A-Za-z0-9&._]+\\.", "", names(df))
  xts(df, order.by = dates)
}

compute_all <- function(x) {
  cl  <- x[, "Close"]; hi <- x[, "High"]; lo <- x[, "Low"]
  vo  <- if ("Volume" %in% names(x)) x[, "Volume"] else NULL
  adj <- if ("Adjusted" %in% names(x)) x[, "Adjusted"] else NULL
  n   <- nrow(x)

  cl  <- na.locf(cl,  na.rm = FALSE); hi <- na.locf(hi, na.rm = FALSE); lo <- na.locf(lo, na.rm = FALSE)
  if (!is.null(vo)) vo <- na.locf(vo, na.rm = FALSE)
  if (!is.null(adj)) adj <- na.locf(adj, na.rm = FALSE)

  add_col <- function(out, obj, name) {
    if (is.null(obj)) return(out)
    if (ncol(obj) == 1) colnames(obj) <- name else names(obj) <- name
    cbind(out, obj)
  }
  safe_ind <- function(expr, min_n) { if (n >= min_n) tryCatch(expr, error = function(e) NULL) else NULL }
  empty <- function() xts(rep(NA, n), order.by = index(cl))
  `%||%` <- function(x, y) if (is.null(x)) y else x

  out <- cl; colnames(out) <- "Close"
  if (!is.null(adj)) out <- add_col(out, adj, "Adjusted")

  out <- add_col(out, safe_ind(SMA(cl, n = 20),  20)  %||% empty(), "SMA20")
  out <- add_col(out, safe_ind(SMA(cl, n = 50),  50)  %||% empty(), "SMA50")
  out <- add_col(out, safe_ind(SMA(cl, n = 200), 200) %||% empty(), "SMA200")
  out <- add_col(out, safe_ind(EMA(cl, n = 12),  12)  %||% empty(), "EMA12")
  out <- add_col(out, safe_ind(EMA(cl, n = 26),  26)  %||% empty(), "EMA26")
  out <- add_col(out, safe_ind(RSI(cl, n = 14),  14)  %||% empty(), "RSI14")

  macd_obj <- safe_ind(MACD(cl, nFast = 12, nSlow = 26, nSig = 9), 26)
  if (!is.null(macd_obj)) {
    out <- add_col(out, macd_obj[, "macd"],   "MACD")
    out <- add_col(out, macd_obj[, "signal"], "MACDsignal")
  } else {
    out <- add_col(out, empty(), "MACD"); out <- add_col(out, empty(), "MACDsignal")
  }

  bb <- safe_ind(BBands(HLC = cbind(hi, lo, cl), n = 20, sd = 2, maType = SMA), 20)
  if (!is.null(bb)) {
    out <- add_col(out, bb[, "up"],   "BBupper")
    out <- add_col(out, bb[, "mavg"], "BBmiddle")
    out <- add_col(out, bb[, "dn"],   "BBlower")
    out <- add_col(out, bb[, "pctB"], "BBpctB")
  } else {
    out <- add_col(out, empty(), "BBupper"); out <- add_col(out, empty(), "BBmiddle")
    out <- add_col(out, empty(), "BBlower"); out <- add_col(out, empty(), "BBpctB")
  }

  atr <- safe_ind(ATR(HLC = cbind(hi, lo, cl), n = 14), 14)
  out <- add_col(out, if (!is.null(atr)) atr[, "atr"] else empty(), "ATR14")

  st <- safe_ind(stoch(HLC = cbind(hi, lo, cl), nFastK = 14, nFastD = 3, nSlowD = 3), 14)
  if (!is.null(st)) {
    out <- add_col(out, st[, "fastK"], "StochFastK")
    out <- add_col(out, st[, "fastD"], "StochFastD")
    out <- add_col(out, st[, "slowD"], "StochSlowD")
  } else {
    out <- add_col(out, empty(), "StochFastK")
    out <- add_col(out, empty(), "StochFastD")
    out <- add_col(out, empty(), "StochSlowD")
  }

  if (!is.null(vo)) {
    out <- add_col(out, safe_ind(OBV(cl, vo), 1) %||% empty(), "OBV")
    typical <- (hi + lo + cl) / 3; tpv <- typical * vo
    rs_tpv <- rollapply(tpv, width = 20, FUN = sum, fill = NA, align = "right")
    rs_vo  <- rollapply(vo,  width = 20, FUN = sum, fill = NA, align = "right")
    out <- add_col(out, rs_tpv / rs_vo, "VWAP20")
  }
  return(out)
}

plot_ticker <- function(ticker, out_path) {
  x <- read_ohlcv(ticker)
  if (is.null(x) || nrow(x) < 100) {
    message("  SKIP plot — insufficient data (", ifelse(is.null(x), 0, nrow(x)), " rows)")
    return(FALSE)
  }
  ind <- compute_all(x)
  first_valid <- 1
  if ("SMA200" %in% colnames(ind) && any(!is.na(ind$SMA200))) {
    first_valid <- which(!is.na(ind$SMA200))[1]
  } else if ("SMA50" %in% colnames(ind) && any(!is.na(ind$SMA50))) {
    first_valid <- which(!is.na(ind$SMA50))[1]
  } else if ("SMA20" %in% colnames(ind) && any(!is.na(ind$SMA20))) {
    first_valid <- which(!is.na(ind$SMA20))[1]
  }
  if (is.na(first_valid)) first_valid <- 1

  ind_sub <- ind[first_valid:nrow(ind), ]
  idx <- index(ind_sub)

  get_vec <- function(name) { if (name %in% colnames(ind_sub)) as.numeric(ind_sub[, name]) else NULL }
  price  <- get_vec("Close"); adj <- get_vec("Adjusted")
  sma20  <- get_vec("SMA20"); sma50 <- get_vec("SMA50"); sma200 <- get_vec("SMA200")
  vwap   <- get_vec("VWAP20")
  bb_up  <- get_vec("BBupper"); bb_lo <- get_vec("BBlower")
  rsi    <- get_vec("RSI14")
  stochK <- get_vec("StochFastK")
  macd   <- get_vec("MACD"); signal <- get_vec("MACDsignal")
  atr    <- get_vec("ATR14"); obv <- get_vec("OBV")
  plot_price <- if (!is.null(adj)) adj else price

  if (length(idx) != length(plot_price) || length(plot_price) == 0) {
    message("  SKIP plot — length mismatch")
    return(FALSE)
  }

  tryCatch({
    png(out_path, width = 1400, height = 1200, res = 100)
    layout(matrix(1:5, ncol = 1), heights = c(4, 2, 2, 2, 2))

    par(mar = c(0, 4, 3, 2))
    ylim1 <- range(c(plot_price, sma20, sma50, sma200, bb_up, bb_lo, vwap), na.rm = TRUE)
    plot(idx, plot_price, type = "l", col = "black", lwd = 1.3, ylim = ylim1, xaxt = "n", ylab = "Price",
         main = paste(ticker, "— SMA | EMA | BBands | RSI | MACD | ATR | VWAP | Stoch | OBV"))
    if (!is.null(sma20) && !all(is.na(sma20)))  lines(idx, sma20,  col = "blue",   lwd = 1)
    if (!is.null(sma50) && !all(is.na(sma50)))  lines(idx, sma50,  col = "orange", lwd = 1)
    if (!is.null(sma200) && !all(is.na(sma200))) lines(idx, sma200, col = "red",    lwd = 1)
    if (!is.null(vwap) && !all(is.na(vwap))) lines(idx, vwap, col = "purple", lwd = 1, lty = "dashed")
    if (!is.null(bb_up) && !all(is.na(bb_up))) {
      lines(idx, bb_up, col = "gray60", lwd = 0.8, lty = "dotted")
      lines(idx, bb_lo, col = "gray60", lwd = 0.8, lty = "dotted")
    }
    legend("topleft", legend = c("Price", "SMA20", "SMA50", "SMA200", "VWAP", "BBands"),
           col = c("black", "blue", "orange", "red", "purple", "gray60"),
           lty = c(1, 1, 1, 1, 2, 3), lwd = c(1.3, 1, 1, 1, 1, 0.8), bty = "n", cex = 0.75, ncol = 3)
    grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")

    par(mar = c(0, 4, 0, 2))
    plot(idx, rsi, type = "l", col = "purple", lwd = 1.2, ylim = c(0, 100), xaxt = "n", ylab = "RSI / Stoch")
    if (!is.null(stochK) && !all(is.na(stochK))) lines(idx, stochK, col = "blue", lwd = 1)
    abline(h = c(20, 80), col = "red", lty = "dashed")
    legend("topleft", legend = c("RSI(14)", "Stoch %K"), col = c("purple", "blue"), lty = 1, lwd = 1.2, bty = "n", cex = 0.75)
    grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")

    par(mar = c(0, 4, 0, 2))
    if (!is.null(macd) && !all(is.na(macd))) {
      ylim3 <- range(c(macd, signal, (macd - signal)), na.rm = TRUE)
      plot(idx, macd, type = "l", col = "blue", lwd = 1.2, ylim = ylim3, xaxt = "n", ylab = "MACD")
      if (!is.null(signal) && !all(is.na(signal))) lines(idx, signal, col = "red", lwd = 1)
      histo <- macd - signal; colors <- ifelse(histo >= 0, "green", "red")
      segments(idx, rep(0, length(idx)), idx, histo, col = colors, lwd = 1.5)
      abline(h = 0, col = "black", lty = "solid")
      legend("topleft", legend = c("MACD", "Signal", "Hist"), col = c("blue", "red", "green"),
             lty = c(1, 1, 1), lwd = c(1.2, 1, 1.5), bty = "n", cex = 0.75)
    } else {
      plot(0, 0, type = "n", xaxt = "n", yaxt = "n", ylab = "MACD")
      text(0, 0, "MACD not available", cex = 1)
    }
    grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")

    par(mar = c(0, 4, 0, 2))
    if (!is.null(atr) && !all(is.na(atr))) {
      plot(idx, atr, type = "l", col = "brown", lwd = 1.2, xaxt = "n", ylab = "ATR(14)")
      legend("topleft", legend = "ATR(14)", col = "brown", lty = 1, bty = "n", cex = 0.75)
    } else {
      plot(0, 0, type = "n", xaxt = "n", yaxt = "n", ylab = "ATR(14)")
      text(0, 0, "ATR not available", cex = 1)
    }
    grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")

    par(mar = c(4, 4, 0, 2))
    if (!is.null(obv) && !all(is.na(obv))) {
      plot(idx, obv, type = "l", col = "darkgreen", lwd = 1.2, xlab = "Date", ylab = "OBV")
      legend("topleft", legend = "OBV", col = "darkgreen", lty = 1, bty = "n", cex = 0.75)
    } else {
      plot(0, 0, type = "n", xlab = "Date", ylab = "OBV", xaxt = "n", yaxt = "n")
      text(0, 0, "OBV not available", cex = 1)
    }
    grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")

    dev.off()
    return(TRUE)
  }, error = function(e) {
    if (names(dev.cur()) != "null device") dev.off()
    message("  PLOT ERROR for ", ticker, ": ", conditionMessage(e))
    return(FALSE)
  })
}

csv_count <- 0; plot_count <- 0
for (t in tickers) {
  message("Processing: ", t)
  x <- read_ohlcv(t)
  if (is.null(x)) { message("  Skipped — file missing"); next }
  ind <- compute_all(x)
  csv_path <- file.path(ind_dir, paste0(t, "_indicators.csv"))
  write.csv(as.data.frame(ind), csv_path)
  csv_count <- csv_count + 1
  message("  CSV saved (", nrow(ind), " rows × ", ncol(ind), " indicators)")
  plot_path <- file.path(plot_dir, paste0("04_indicators_", gsub("\\.", "_", t), ".png"))
  ok <- plot_ticker(t, plot_path)
  if (ok) { plot_count <- plot_count + 1; message("  Plot saved") }
}

message("\n============================================================")
message("Done — ", csv_count, " indicator CSVs written to ", ind_dir)
message("Done — ", plot_count, " indicator plots written to ", plot_dir)
message("============================================================")
