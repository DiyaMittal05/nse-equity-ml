#!/usr/bin/env Rscript
# 03_eda.R — EDA plots for the universe panel

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

clean_dir <- CLEAN_DIR
plot_dir  <- PLOT_DIR
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

# ---- Read aligned data ----
prices <- read.csv(file.path(clean_dir, "close_prices_aligned.csv"), stringsAsFactors = FALSE)
prices_dates <- as.Date(prices[[1]]); prices <- prices[, -1]
prices_xts <- xts(prices, order.by = prices_dates)

returns <- read.csv(file.path(clean_dir, "daily_log_returns.csv"), stringsAsFactors = FALSE)
ret_dates <- as.Date(returns[[1]]); returns <- returns[, -1]
returns_xts <- xts(returns, order.by = ret_dates)

# ---- Rebase to 100 ----
rebased <- prices_xts
for (t in colnames(rebased)) {
  col <- rebased[, t]
  fv <- col[which(!is.na(col))[1]]
  rebased[, t] <- col / fv * 100
}

# ---- Panel 1: Normalised prices ----
valid <- colnames(rebased)[colSums(!is.na(rebased)) > 100]
plot_data <- rebased[, valid]

png(file.path(plot_dir, "01_normalised_prices.png"), width = 1200, height = 700, res = 100)
par(mar = c(4, 4, 3, 1))
matplot(index(plot_data), coredata(plot_data), type = "l", lty = 1, lwd = 1.2,
        log = "y", xlab = "Date", ylab = "Rebased Price (log scale)",
        main = paste(UNIVERSE_LABEL, "— Normalised Prices (rebased to 100)"))
legend("topleft", legend = valid, col = 1:length(valid), lty = 1, lwd = 1.2, cex = 0.7, ncol = 3, bty = "n")
grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")
dev.off()
cat("Wrote 01_normalised_prices.png\n")

# ---- Panel 2: Return distributions (top 12) ----
ret_mat <- coredata(returns_xts)
plot_cols <- colnames(ret_mat)[order(-colSums(!is.na(ret_mat)))][1:min(12, ncol(ret_mat))]
n <- length(plot_cols)
rows <- ceiling(n / 4); cols <- 4
png(file.path(plot_dir, "02_return_distributions.png"), width = 1400, height = 1000, res = 100)
par(mfrow = c(rows, cols), mar = c(3, 3, 2, 1))
for (t in plot_cols) {
  v <- na.omit(ret_mat[, t])
  if (length(v) < 30) { plot.new(); next }
  hist(v, breaks = 60, main = t, xlab = "Daily log-return", col = "steelblue", border = "white")
  abline(v = mean(v), col = "red", lwd = 1.5)
  legend("topright", legend = paste("SD:", round(sd(v)*100, 2), "%"),
         bty = "n", cex = 0.7)
}
dev.off()
cat("Wrote 02_return_distributions.png\n")

# ---- Panel 3: Rolling volatility (5 majors) ----
majors <- c("SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS")
avail <- intersect(majors, colnames(returns_xts))
if (length(avail) > 0) {
  ret_sub <- returns_xts[, avail]
  roll_vol <- rollapply(ret_sub, width = 30, FUN = sd, fill = NA, align = "right") * sqrt(252) * 100
  roll_vol <- roll_vol[-(1:29), ]
  png(file.path(plot_dir, "03_rolling_volatility.png"), width = 1200, height = 700, res = 100)
  par(mar = c(4, 4, 3, 1))
  ylim_vol <- range(coredata(roll_vol), na.rm = TRUE)
  plot(index(roll_vol), coredata(roll_vol)[, 1], type = "l", col = 1, lwd = 1.2,
       ylim = ylim_vol, xlab = "Date", ylab = "Annualised Volatility (%)",
       main = paste(UNIVERSE_LABEL, "— 30-Day Rolling Volatility"))
  for (i in 2:ncol(roll_vol)) lines(index(roll_vol), coredata(roll_vol)[, i], col = i, lwd = 1.2)
  legend("topleft", legend = avail, col = 1:ncol(roll_vol), lty = 1, lwd = 1.2, cex = 0.8, bty = "n")
  grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")
  dev.off()
  cat("Wrote 03_rolling_volatility.png\n")
}

cat("EDA complete. Plots in:", plot_dir, "\n")
