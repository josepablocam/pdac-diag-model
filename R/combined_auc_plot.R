#!/usr/bin/env Rscript

library(ggplot2)
library(optparse)

ORDERED_EXPERIMENTS <- c(
    "BIDMC-Test",
    "PHC-Test",
    "PHC-Retrained"
)


plot_auc <- function(dat) {
    dat$cutoff_text <- paste(dat$cutoff, "Days")
    dat$experiment <- factor(dat$experiment, ORDERED_EXPERIMENTS)
    ggplot(data=dat, aes(x=experiment, y=roc_auc, fill=label)) +
    geom_bar(position="dodge", stat="identity") +
    facet_wrap(~cutoff_text, ncol=1) +
    geom_errorbar(aes(ymin=roc_auc_ci_lb, ymax=roc_auc_ci_ub), position="dodge") +
    labs(x="Experiment", fill="Model", y="AUC") +
    coord_cartesian(ylim=c(0.5, 1.0)) +
    theme(legend.position="bottom")
}


option_list = list(
  make_option(c("--input_path"), type="character",
              help="Input file with AUC information"),
	make_option(c("--output_path"), type="character",
              help="Path to dump plot")
);

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$input_path) || is.null(opt$output_path)) {
  print_help(opt_parser)
} else {
  theme_set(theme_grey(base_size = 18))
  dat <- read.csv(opt$input_path, stringsAsFactors=FALSE)
  p <- plot_auc(dat)
  ggsave(opt$output_path, p)
}
