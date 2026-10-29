# config.R — shared settings for the PSU / NBFC R pipeline
#
# Every script is run as:   Rscript r_analysis/<script>.R psu|nbfc
# and writes into Data_analysis_<universe>/ at the repo root (data/, cleaned/, plots/, indicators/).

UNIVERSES <- list(
  psu = list(
    label = "PSU Banks",
    tickers = c("SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS",
                "UNIONBANK.NS", "INDIANB.NS", "BANKINDIA.NS", "CENTRALBK.NS",
                "MAHABANK.NS", "IOB.NS", "UCOBANK.NS", "PSB.NS")
  ),
  nbfc = list(
    label = "NBFCs",
    tickers = c("BAJFINANCE.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS", "M&MFIN.NS",
                "LTF.NS", "SBICARD.NS", "MUTHOOTFIN.NS", "IIFL.NS",
                "TATACAP.NS", "ABCAPITAL.NS", "MUTHOOTMF.NS",
                "FEDFINA.NS", "MANAPPURAM.NS", "CREDITACC.NS", "PAISALO.NS",
                "MONEYBOXX.NS", "UGROCAP.NS", "SBFC.NS", "INDOSTAR.NS",
                "MASFIN.NS", "SUNDARMFIN.NS", "EDELWEISS.NS", "BAIDFIN.NS",
                "JIOFIN.NS", "FUSION.NS", "FIVESTAR.NS", "POONAWALLA.NS",
                "AFSL.NS", "PFS.NS", "NIYOGIN.BO", "PNBGILTS.NS")
  )
)

.args <- commandArgs(trailingOnly = TRUE)
UNIVERSE <- if (length(.args) >= 1) tolower(.args[1]) else "psu"
if (!UNIVERSE %in% names(UNIVERSES)) {
  stop("Unknown universe '", UNIVERSE, "'. Use one of: ", paste(names(UNIVERSES), collapse = ", "))
}

UNIVERSE_LABEL <- UNIVERSES[[UNIVERSE]]$label
TICKERS        <- UNIVERSES[[UNIVERSE]]$tickers

# SCRIPT_DIR is set by the calling script before sourcing this file
REPO_ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."))
BASE_DIR  <- file.path(REPO_ROOT, paste0("Data_analysis_", UNIVERSE))
RAW_DIR   <- file.path(BASE_DIR, "data")
CLEAN_DIR <- file.path(BASE_DIR, "cleaned")
PLOT_DIR  <- file.path(BASE_DIR, "plots")
IND_DIR   <- file.path(BASE_DIR, "indicators")

cat("Universe:", UNIVERSE_LABEL, "->", BASE_DIR, "\n")
