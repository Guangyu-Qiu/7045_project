# Defaults: 50 subjects, seed 648 and serial execution (one worker).
# Binary MAE/RMSE compare probabilities with observed 0/1 outcomes.
# The field `bias` stores prediction errors, not repeated-simulation bias.

# ---- simulate-data ----
# Input: n = 50, seed 648, predictor distributions and GLM coefficients.
# Output: cont, binary and known probabilities p; data remain in memory.
set.seed(648)

n <- 50L
X <- cbind(
  intercept = rep(1, n),
  x1 = rnorm(n, mean = 0, sd = 1),
  x2 = rnorm(n, mean = 5, sd = 5),
  x3 = rpois(n, lambda = 3),
  x4 = rbinom(n, size = 1, prob = 0.3),
  x5 = runif(n, min = 0, max = 10)
)

beta_cont <- c(1, 1, -0.5, 2, -2, 0.5)
cont <- data.frame(X, Y = drop(X %*% beta_cont) + rnorm(n, 0, 1))

# Retain the original coefficients and unstandardized predictors.
beta_binary <- c(-1, 0.3, -0.1, 0.3, -0.5, 0.1)
p <- plogis(drop(X %*% beta_binary))
binary <- data.frame(X, Y = rbinom(n, size = 1, prob = p))

# Keep truth outside binary so it cannot accidentally enter the fitted model.
# No existing CSV files are overwritten by this standalone example.
print(table(binary$Y))

# ---- predictor-summary ----
# Input: the shared predictors in cont.
# Output: predictor_summary with mean, SD and range.
predictor_names <- paste0("x", 1:5)
predictor_summary <- data.frame(
  Predictor = predictor_names,
  Mean = vapply(cont[predictor_names], mean, numeric(1)),
  SD = vapply(cont[predictor_names], sd, numeric(1)),
  Minimum = vapply(cont[predictor_names], min, numeric(1)),
  Maximum = vapply(cont[predictor_names], max, numeric(1)),
  row.names = NULL
)
print(predictor_summary, digits = 3, row.names = FALSE)

# ---- predictor-distributions ----
# Input: the five shared predictors.
# Output: histograms and count plots of their distributions.
predictor_long <- stack(cont[predictor_names])
print(
ggplot2::ggplot(mapping = ggplot2::aes(x = values)) +
  ggplot2::geom_histogram(
    data = subset(predictor_long, ind %in% c("x1", "x2", "x5")),
    bins = 10, fill = "steelblue", color = "white"
  ) +
  ggplot2::geom_bar(
    data = subset(predictor_long, ind %in% c("x3", "x4")),
    width = 0.8, fill = "steelblue", color = "white"
  ) +
  ggplot2::facet_wrap(~ ind, scales = "free", ncol = 3) +
  ggplot2::labs(x = "Observed value", y = "Count") +
  ggplot2::theme_minimal(base_size = 12)
)

# ---- outcome-summary ----
# Input: cont and binary.
# Output: outcome_summary for both responses.
outcome_summary <- data.frame(
  Dataset = c("Continuous", "Binary"),
  N = c(nrow(cont), nrow(binary)),
  Mean = c(mean(cont$Y), mean(binary$Y)),
  SD = c(sd(cont$Y), sd(binary$Y)),
  Minimum = c(min(cont$Y), min(binary$Y)),
  Maximum = c(max(cont$Y), max(binary$Y))
)
print(outcome_summary, digits = 3, row.names = FALSE)

# ---- continuous-outcome ----
# Input: cont$Y.
# Output: a continuous-outcome histogram.
print(
ggplot2::ggplot(cont, ggplot2::aes(x = Y)) +
  ggplot2::geom_histogram(bins = 10, fill = "steelblue", color = "white") +
  ggplot2::labs(title = "Continuous outcome", x = "Y", y = "Count") +
  ggplot2::theme_minimal(base_size = 12)
)

# ---- binary-outcome ----
# Input: binary$Y.
# Output: counts of outcomes 0 and 1.
print(
ggplot2::ggplot(binary, ggplot2::aes(x = factor(Y, levels = c(0, 1)))) +
  ggplot2::geom_bar(fill = "steelblue", width = 0.6) +
  ggplot2::geom_text(ggplot2::aes(label = ggplot2::after_stat(count)),
                     stat = "count", vjust = -0.4) +
  ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.12))) +
  ggplot2::labs(title = "Binary outcome", x = "Y", y = "Count") +
  ggplot2::theme_minimal(base_size = 12)
)

# ---- fit-function ----
# Input: formula, training data, family function and link.
# Output: fit_fun(), returning a named coefficient vector.
#' Fit a GLM and retain only its named coefficient vector.
#' @param formula A model formula with a single numeric response.
#' @param data A data frame containing the formula variables.
#' @param family A GLM family function, for example gaussian or binomial.
#' @param link A character string naming the link.
#' @return A named numeric coefficient vector.
fit_fun <- function(formula, data, family = gaussian, link = "identity") {
  if (!is.function(family)) {
    stop("family must be a function, such as gaussian or binomial.")
  }
  if (!is.character(link) || length(link) != 1L || is.na(link)) {
    stop("link must be one string, such as 'identity' or 'logit'.")
  }

  mf <- stats::model.frame(formula, data = data, na.action = stats::na.fail)
  X <- stats::model.matrix(attr(mf, "terms"), data = mf)
  y <- stats::model.response(mf)
  if (!is.numeric(y) || !is.null(dim(y))) {
    stop("Use a single numeric response (0/1 for the binary example).")
  }
  if (any(!is.finite(X)) || any(!is.finite(y))) {
    stop("Model variables must be finite and contain no missing values.")
  }

  fit <- stats::glm.fit(x = X, y = y, family = family(link = link))
  if (!isTRUE(fit$converged)) {
    stop("The GLM did not converge. Inspect the data and model specification.")
  }
  if (fit$rank < ncol(X) || any(!is.finite(fit$coefficients))) {
    stop("The model has coefficients that cannot be estimated uniquely.")
  }
  fit$coefficients
}

# ---- predictions-and-s3 ----
# Input: omitted-row coefficients, original data and model settings.
# Output: predict_fun() and S3 print, summary and plot methods.
#' Make row-wise LOO predictions and assemble a result object.
#' @param beta Matrix whose ith row contains coefficients fitted without row i.
#' @param dat The original data frame, in its original row order.
#' @param formula The formula used in every fit.
#' @param family The family function used in every fit.
#' @param link The link string used in every fit.
#' @return An object of class loo_results.
predict_fun <- function(beta, dat, formula, family = gaussian,
                        link = "identity") {
  mf <- stats::model.frame(formula, data = dat, na.action = stats::na.fail)
  X <- stats::model.matrix(attr(mf, "terms"), data = mf)
  y <- stats::model.response(mf)

  if (!is.matrix(beta) || nrow(beta) != nrow(X) ||
      ncol(beta) != ncol(X) ||
      !setequal(colnames(beta), colnames(X))) {
    stop("beta must have one row per observation and matching model columns.")
  }
  beta <- beta[, colnames(X), drop = FALSE]
  family_spec <- family(link = link)
  eta <- rowSums(X * beta)
  pred <- family_spec$linkinv(eta)
  if (any(!is.finite(pred))) {
    stop("The model produced non-finite predictions.")
  }
  bias <- pred - y

  results <- list(
    formula = formula,
    family = family_spec,
    nrow = nrow(dat),
    ncol = ncol(dat),
    coefficients = beta,
    observed = y,
    predicted = pred,
    bias = bias,
    mae = mean(abs(bias)),
    rmse = sqrt(mean(bias^2))
  )
  class(results) <- "loo_results"
  results
}

#' @param x A loo_results object.
#' @param ... Additional method arguments.
#' @return The input object, invisibly.
print.loo_results <- function(x, ...) {
  cat("Leave-one-out prediction metrics\n")
  cat(sprintf("MAE:  %.6f\nRMSE: %.6f\n", x$mae, x$rmse))
  invisible(x)
}

#' @param object A loo_results object.
#' @param ... Additional method arguments.
#' @return The input object, invisibly, after displaying its summary.
summary.loo_results <- function(object, ...) {
  cat("Formula:", paste(deparse(object$formula), collapse = " "), "\n")
  cat("Data:", object$nrow, "rows and", object$ncol, "columns\n")
  cat("Family:", object$family$family,
      "| Link:", object$family$link, "\n")
  print(object)
  invisible(object)
}

# R already provides the plot() generic for S3 dispatch.
#' @param x A loo_results object.
#' @param bins Number of histogram bins.
#' @param ... Reserved for additional arguments.
#' @return A ggplot histogram of predicted minus observed errors.
plot.loo_results <- function(x, bins = 10L, ...) {
  ggplot2::ggplot(data.frame(bias = x$bias), ggplot2::aes(x = bias)) +
    ggplot2::geom_histogram(bins = bins, color = "white", fill = "steelblue") +
    ggplot2::labs(
      title = "LOO prediction errors",
      x = "Predicted - observed", y = "Count"
    ) +
    ggplot2::theme_minimal()
}

# ---- run-loo ----
# Input: formula, full data, family, link and worker count.
# Output: run_loo(), returning predictions, errors, MAE and RMSE.
#' Run one-row-at-a-time LOO validation.
#' @param formula A model formula.
#' @param data The complete data frame.
#' @param family A family function, for example gaussian or binomial.
#' @param link A link string.
#' @param workers Number of workers. The default 1 runs serially.
#' @return An object of class loo_results.
run_loo <- function(formula, data, family = gaussian, link = "identity",
                    workers = 1L) {
  if (!is.data.frame(data) || nrow(data) < 3L) {
    stop("data must be a data frame with at least three rows.")
  }
  if (!is.numeric(workers) || length(workers) != 1L ||
      !is.finite(workers) || workers < 1 || workers != floor(workers)) {
    stop("workers must be a positive integer.")
  }

  indices <- seq_len(nrow(data))
  if (workers == 1L) {
    res <- lapply(indices, function(i) {
      fit_fun(formula, data[-i, , drop = FALSE], family, link)
    })
  } else {
    cl <- parallel::makePSOCKcluster(as.integer(workers))
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterExport(
      cl, c("fit_fun", "formula", "data", "family", "link"),
      envir = environment()
    )
    res <- parallel::parLapplyLB(cl, indices, function(i) {
      fit_fun(formula, data[-i, , drop = FALSE], family, link)
    })
  }

  # Avoid silently combining coefficient vectors with different model columns.
  same_columns <- vapply(res, function(b) {
    identical(names(b), names(res[[1L]]))
  }, logical(1))
  if (!all(same_columns)) {
    stop("The model columns changed between LOO fits.")
  }
  beta <- do.call(rbind, res)
  predict_fun(beta, data, formula, family, link)
}

# ---- examples ----
# Input: the two simulated datasets and the five-predictor formula.
# Output: cont_result, binary_result, printed metrics and error plots.
model_formula <- Y ~ x1 + x2 + x3 + x4 + x5

cont_result <- run_loo(model_formula, cont, gaussian, "identity")
binary_result <- run_loo(model_formula, binary, binomial, "logit")

print(cont_result)
summary(binary_result)
print(plot(cont_result))
print(plot(binary_result))

# ---- binary-diagnostics ----
# Input: LOO probabilities, observed binary outcomes and known simulation truth.
# Output: observed-outcome scores, oracle expected risk and probability recovery.
# Observed-outcome MAE/RMSE stay on the original assignment's target.
# Brier score is mean squared probability error against observed 0/1 outcomes.
# The intercept-only benchmark also excludes each subject's own outcome.
null_probability <- (sum(binary$Y) - binary$Y) / (nrow(binary) - 1L)
score_predictions <- function(predicted, observed) {
  error <- predicted - observed
  c(MAE = mean(abs(error)), RMSE = sqrt(mean(error^2)),
    Brier = mean(error^2))
}
binary_scores <- data.frame(
  Predictor = c("LOO logistic", "LOO intercept-only", "Known probability (realized)"),
  rbind(score_predictions(binary_result$predicted, binary$Y),
        score_predictions(null_probability, binary$Y),
        score_predictions(p, binary$Y)),
  row.names = NULL
)
print(binary_scores, digits = 4, row.names = FALSE)

# For Y ~ Bernoulli(p), E[(p-Y)^2 | p] = p(1-p).
# sqrt(mean(p*(1-p))) is the square root of expected MSE, conditional on X;
# it is not a strict lower bound for RMSE in one realized random sample.
# The oracle MAE below evaluates prediction p; p need not minimize MAE.
oracle_risk <- data.frame(
  Scenario = "Original model",
  Oracle_expected_MAE = 2 * mean(p * (1 - p)),
  Oracle_root_expected_MSE = sqrt(mean(p * (1 - p)))
)
print(oracle_risk, digits = 4, row.names = FALSE)

# A separate simulation-only target: recovering p rather than predicting Y.
# These are not replacements for the observed-outcome scores above.
probability_error <- binary_result$predicted - p
probability_recovery <- data.frame(
  Target = "Known generating probability p",
  MAE = mean(abs(probability_error)),
  RMSE = sqrt(mean(probability_error^2))
)
print(probability_recovery, digits = 4, row.names = FALSE)

# ---- verification ----
# Input: this script's results and functions.
# Output: independent-refit checks and default/serial agreement checks.
# Independent glm()/predict() refits check the entire held-out-row workflow.
reference_predictions <- function(data, family, link) {
  vapply(seq_len(nrow(data)), function(i) {
    fit <- stats::glm(model_formula, data = data[-i, , drop = FALSE],
                      family = family(link = link))
    stopifnot(isTRUE(fit$converged), all(is.finite(stats::coef(fit))))
    as.numeric(stats::predict(fit, newdata = data[i, , drop = FALSE],
                              type = "response"))
  }, numeric(1))
}
cont_reference <- reference_predictions(cont, gaussian, "identity")
binary_reference <- reference_predictions(binary, binomial, "logit")
cont_serial <- run_loo(model_formula, cont, gaussian, "identity", workers = 1L)
binary_serial <- run_loo(model_formula, binary, binomial, "logit", workers = 1L)
stopifnot(
  isTRUE(all.equal(unname(cont_result$predicted), cont_reference)),
  isTRUE(all.equal(unname(binary_result$predicted), binary_reference)),
  isTRUE(all.equal(cont_result, cont_serial)),
  isTRUE(all.equal(binary_result, binary_serial)),
  isTRUE(all.equal(binary_result$mae, mean(abs(binary_reference - binary$Y)))),
  isTRUE(all.equal(binary_result$rmse,
                   sqrt(mean((binary_reference - binary$Y)^2)))),
  all(binary_result$predicted >= 0 & binary_result$predicted <= 1)
)
cat("Verification passed: independent refits, metrics and default/serial agreement.\n")
