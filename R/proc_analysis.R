#!/usr/bin/env Rscript

library(pROC)
library(combinat)
library(optparse)
library(ggplot2)


MODEL_NAMES <- c("lr"="LR", "baecker"="LR-Clinical (baseline)", "nn"="NN")

plot_results <- function(dat) {
    dat$model1 <- MODEL_NAMES[dat$model1]
    dat$model2 <- MODEL_NAMES[dat$model2]
    is_significant_str <- ifelse(dat$is_significant, "(*)", "")
    dat$statistic_formatted <- paste(round(dat$statistic, 2), is_significant_str)

    p <- ggplot(data=dat, aes(x=model1, y=model2, fill=statistic)) +
        facet_wrap(~cutoff) +
        # heat map
        geom_tile(color = "gray") +
        scale_fill_gradient(low = "white", high = "palegreen3") +
        # add value in each square (asterisk if sig)
        geom_text(aes(label=statistic_formatted), size=4.5) +
        # make cells squared
        coord_fixed() +
        labs(x="Model 1", y="Model 2", fill="deLong ") +
        theme(
          panel.background = element_blank()
        ) +
        theme(legend.position="bottom")

    p
}


compare_results <- function(input_path, output_path, alpha) {
    theme_set(theme_grey(base_size = 18))
    dat <- read.csv(input_path, stringsAsFactors=FALSE)

    # make sure logical
    dat$ytrue <- dat$ytrue == "True"

    cutoffs <- unique(dat$cutoff)
    models <- unique(dat$model)

    # bonferroni correction
    ncomparisons <- 1

    # roc.test is pairwaise comparisons
    model_comparisons <- combn(models, 2)
    num_model_comps <- ncol(model_comparisons)

    acc <- list()

    for (cutoff in cutoffs) {
        dat_cutoff <- dat[dat$cutoff == cutoff, ]

        for(comb_ix in seq(num_model_comps)) {
            model_comp <- model_comparisons[, comb_ix]
            model1 <- model_comp[[1]]
            model2 <- model_comp[[2]]

            dat_model1 <- dat_cutoff[dat_cutoff$model == model1, ]
            dat_model2 <- dat_cutoff[dat_cutoff$model == model2, ]

            # can use either dataframe since same
            ytrue <- dat_model1$ytrue
            probs1 <- dat_model1$yprob
            probs2 <- dat_model2$yprob

            result <- roc.test(ytrue, probs1, probs2, method="delong", paired=TRUE)
            result_df <- data.frame(
              cutoff=cutoff,
              model1=model1,
              model2=model2,
              statistic=result$statistic,
              pvalue=result$p.value
            )
            acc[[ncomparisons]] <- result_df
            ncomparisons <- ncomparisons + 1
        }
    }


    final_df <- do.call(rbind, acc)
    final_df$model1 <- as.character(final_df$model1)
    final_df$model2 <- as.character(final_df$model2)
    # adjust p.values for repeated comparisons
    print(paste("Performed", ncomparisons, "comparisons, adjusting pvals"))
    final_df$pvalue <- final_df$pvalue * ncomparisons
    final_df$is_significant <- final_df$pvalue < alpha
    final_df$alpha <- alpha
    print("Results")
    print(final_df)
    write.csv(final_df, paste0(output_path, ".csv"), row.names=FALSE)

    p <- plot_results(final_df)
    ggsave(paste0(output_path, ".pdf"), p)
}


option_list = list(
  make_option(c("--input_path"), type="character",
              help="Input file with AUC information"),
  make_option(c("--alpha"), type="numeric", default=0.05,
              help="P-value cutoff"),
	make_option(c("--output_path"), type="character",
              help="Path to dump output csv and pdf with results")
);

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$input_path) || is.null(opt$output_path)) {
  print_help(opt_parser)
} else {
  compare_results(opt$input_path, opt$output_path, opt$alpha)
}
