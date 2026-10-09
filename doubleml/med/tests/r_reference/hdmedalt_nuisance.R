# Nuisance part of causalweight:::hdmedalt (fewsplits=FALSE, MLmethod="parametric"), same lines as in
# causalweight/R/functions.R, writing the folds and the predictions of E(Y|M,X,D=d) and E[E(Y|M,X,D=d)|D=1-d,X].
#
# Produced with R 4.5.0, SuperLearner 2.0.40 and causalweight 1.1.5 (all on CRAN).
# Regenerate r_preds.csv and r_folds.csv from data.csv with, from this folder:
#   Rscript hdmedalt_nuisance.R .
suppressMessages({library(SuperLearner); library(causalweight)})
MLfunct <- causalweight:::MLfunct
args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(file.path(args[1], "data.csv"))
y <- dat$y; d <- dat$d; m <- dat$m; x <- as.matrix(dat[, grep("^X", names(dat))])
ybin <- 1 * (length(unique(y)) == 2 & min(y) == 0 & max(y) == 1)
xm <- cbind(x, m)
stepsize <- ceiling((1 / 3) * length(d))
nobs <- min(3 * stepsize, length(d)); set.seed(1); idx <- sample(nobs)
sample1 <- idx[1:stepsize]; sample2 <- idx[(stepsize + 1):(2 * stepsize)]; sample3 <- idx[(2 * stepsize + 1):nobs]
preds <- c(); folds <- c()
for (i in 1:3) {
  if (i == 1) {tesample <- sample1; musample <- sample2; deltasample <- sample3}
  if (i == 2) {tesample <- sample3; musample <- sample1; deltasample <- sample2}
  if (i == 3) {tesample <- sample2; musample <- sample3; deltasample <- sample1}
  xm_df <- data.frame(xm); x_df <- data.frame(x)
  dtrte <- d[deltasample]; xtrte <- x_df[deltasample, ]
  eymx1 <- MLfunct(y = y[musample[d[musample] == 1]], x = xm_df[musample[d[musample] == 1], ], MLmethod = "parametric", ybin = ybin)
  eymx1te <- predict(eymx1, xm_df[tesample, ], onlySL = TRUE)$pred
  eymx1trte <- predict(eymx1, xm_df[deltasample, ], onlySL = TRUE)$pred
  regweymx1 <- MLfunct(y = eymx1trte[dtrte == 0], x = xtrte[dtrte == 0, ], MLmethod = "parametric", ybin = 0)
  regweymx1te <- predict(regweymx1, x_df[tesample, ], onlySL = TRUE)$pred
  eymx0 <- MLfunct(y = y[musample[d[musample] == 0]], x = xm_df[musample[d[musample] == 0], ], MLmethod = "parametric", ybin = ybin)
  eymx0te <- predict(eymx0, xm_df[tesample, ], onlySL = TRUE)$pred
  eymx0trte <- predict(eymx0, xm_df[deltasample, ], onlySL = TRUE)$pred
  regweymx0 <- MLfunct(y = eymx0trte[dtrte == 1], x = xtrte[dtrte == 1, ], MLmethod = "parametric", ybin = 0)
  regweymx0te <- predict(regweymx0, x_df[tesample, ], onlySL = TRUE)$pred
  preds <- rbind(preds, data.frame(fold = i, unit = tesample - 1, eymx1te = as.numeric(eymx1te), regweymx1te = as.numeric(regweymx1te),
                                   eymx0te = as.numeric(eymx0te), regweymx0te = as.numeric(regweymx0te)))
  folds <- rbind(folds, data.frame(fold = i, role = "test", unit = tesample - 1),
                 data.frame(fold = i, role = "A", unit = musample - 1), data.frame(fold = i, role = "B", unit = deltasample - 1))
}
write.csv(preds, file.path(args[1], "r_preds.csv"), row.names = FALSE)
write.csv(folds, file.path(args[1], "r_folds.csv"), row.names = FALSE)
cat("done", nrow(preds), "\n")
