#!/usr/bin/env Rscript

library(boot)
library(ggplot2)
library(ggsignif)
library(optparse)
library(reshape2)
library(plyr)


RISK_GROUPS <- c("Low", "Intermediate", "High")
# RISK_GROUPS <- c("p90", "p95", "p99", "p995", "p999")
DEFAULT_MISSING_VALUE <- 0
ITER_COUNT <- 0

reset_iter_count <- function() {
  ITER_COUNT <<- 0
}

increment_iter_count <- function() {
  ITER_COUNT <<- ITER_COUNT + 1
}

print_iter_count <- function(every) {
  if ((ITER_COUNT %% every) == 0) {
    print(paste(ITER_COUNT, "iterations"))
  }
}


# Computed within a cutoff
compute_stats <- function(dat, ixs) {
  increment_iter_count()
  print_iter_count(10)

  dat <- dat[ixs, ]

  dat_summary <- ddply(dat, .(risk_group), summarize,
    rate=mean(ytrue),
    patients=length(ytrue),
    cancer=sum(ytrue)
  )
  dat_summary$controls <- dat_summary$patients - dat_summary$cancer

  dat_summary$total_cancer <- sum(dat_summary$cancer)
  dat_summary$total_controls <- sum(dat_summary$controls)
  dat_summary$total_patients <- sum(dat_summary$patients)

  dat_summary$fraction_population <- dat_summary$patients / dat_summary$total_patients
  dat_summary$sensitivity <- dat_summary$cancer / dat_summary$total_cancer
  dat_summary$specificity <- 1 - (dat_summary$controls / dat_summary$total_controls)
  dat_summary$prevalence <- dat_summary$total_cancer / dat_summary$total_patients

  flat_dat_summary <- melt(dat_summary, id.vars=c("risk_group"))
  flat_dat_summary$full_label <- paste(flat_dat_summary$risk_group, flat_dat_summary$variable, sep=".")

  stats_computed <- flat_dat_summary$value
  names(stats_computed) <- flat_dat_summary$full_label

  unique_stats <- unique(flat_dat_summary$variable)
  expected_labels <- c()
  for(g in RISK_GROUPS) {
    for(v in unique_stats) {
      expected_labels <- c(expected_labels, paste(g, v, sep="."))
    }
  }

  # adjust in case none of a particular group
  values <- stats_computed[expected_labels]
  names(values) <- expected_labels
  is_missing <- is.na(values)
  missing_stats <- names(which(is_missing))

  if (length(missing_stats) > 0) {
      print("Missing statistics")
      print(missing_stats)
  }

  values[is_missing] <- DEFAULT_MISSING_VALUE

  values
}

assemble_ci_dataframe <- function(boot_out, conf=0.95) {
    observed <- boot_out$t0
    n <- length(observed)
    labels <- names(observed)
    result <- list()

    for(i in seq_along(labels)) {
        # compute empirical bootstrap for each statistic
        label <- labels[[i]]
        name_parts <- strsplit(label, "\\.")[[1]]
        risk_group <- name_parts[[1]]
        stat <- name_parts[[2]]

        ci_out <- boot.ci(boot_out, conf=conf, index=i, type="perc")

        if(!is.null(ci_out)) {
            val <- ci_out$t0[[1]]
            lb <- ci_out$percent[4]
            ub <- ci_out$percent[5]
        } else {
            print(paste(label, "no CI, since all equal"))
            val <- observed[[label]]
            lb <- val
            ub <- val
        }

        entry <- data.frame(
            risk_group=risk_group,
            stat=stat,
            value=val,
            lb=lb,
            ub=ub
        )
        result[[i]] <- entry

    }
    dat <- do.call(rbind, result)
    dat
}

compute_fisher_test <- function(dat, comparisons) {
   cutoffs <- unique(dat$cutoff)
   results <- list()
   i <- 1
  for(cutoff in cutoffs) {
      cts <- ddply(dat[dat$cutoff == cutoff, ],
        .(risk_group), summarise, cancer=sum(ytrue), no_cancer=sum(!ytrue)
      )
      for (comp in comparisons) {
        split_comp <- strsplit(comp, "-")[[1]]
        cts_comp <- cts[cts$risk_group %in% split_comp, c("cancer", "no_cancer")]
        cts_comp_mat <- as.matrix(cts_comp)
        res <- fisher.test(cts_comp_mat, alternative="two.sided")
        row <- data.frame(cutoff=cutoff, group1=split_comp[[1]], group2=split_comp[[2]], pvalue=res$p.value)
        results[[i]] <- row
        i <- i + 1
      }
  }
    results_df <- do.call(rbind, results)
    results_df
}

annotate_significance <- function(pval, mapping, default_label) {
  min_label <- which.min(mapping[which(mapping > pval)])
  if(length(min_label) == 0) {
    default_label
  } else {
    names(min_label)
  }
}

build_plots <- function(dat, raw_dat, significance_comparisons) {
  dat$cutoff_label <- paste(dat$cutoff, "Days")
  rate_data <- dat[dat$stat == "rate", ]
  preval_data <- dat[dat$stat == "prevalence", ]
  frac_data <- dat[dat$stat == "fraction_population", ]

  # rate of cancer plot
  rate_plot <- ggplot(rate_data, aes(x=risk_group, y=value)) +
    facet_wrap(~cutoff_label) +
    geom_bar(position="dodge", stat="identity") +
    geom_errorbar(aes(ymin=lb, ymax=ub), position="dodge")

  # add in prevalence
  rate_plot <- rate_plot +
    geom_hline(data=preval_data,
      aes(yintercept=value, color="Prevalence"), linetype="dashed") +
    labs(
      x="Group",
      y="Fraction with Cancer"
    ) + theme(legend.title=element_blank()) +
      theme(legend.position="bottom")


  # add in statistical significance markers
  # we perform comparisons indicated
  num_comparisons <- length(unique(dat$cutoff)) * length(significance_comparisons)
  # compute fisher.test for these
  # and compute bonferroni adjustment
  significance_levels <- c("*"=0.05, "**"=0.01, "***"=0.001) / num_comparisons
  # so that ggsignif maps to correct pvalue marker post comparison
  fisher_results <- compute_fisher_test(raw_dat, significance_comparisons)
  fisher_results[["annotation"]]<- sapply(
    fisher_results$pvalue,
    function(v) annotate_significance(v, significance_levels, "ns")
  )
  y_pos_annot <- max(rate_data$ub)
  fisher_results$y_position <- y_pos_annot
  fisher_results$cutoff_label <- paste(fisher_results$cutoff, "Days")

  rate_plot <- rate_plot +
    geom_signif(data=fisher_results, aes(
      cutoff_label=cutoff_label,
      xmin=group1,
      xmax=group2,
      annotations=annotation,
      y_position=y_position + y_position * 0.1
    ), manual=TRUE,
    step_increase = 0.05
  )
  rate_plot <- rate_plot + theme(axis.text.x = element_text(angle = 45, vjust = 0.9, hjust=1))


  # fraction of pop in groups
  frac_plot <- ggplot(frac_data, aes(x=risk_group, y=value)) +
    facet_wrap(~cutoff_label) +
    geom_bar(position="dodge", stat="identity") +
    geom_errorbar(aes(ymin=lb, ymax=ub), position="dodge") +
    labs(x="Group", y="Fraction of Population")
  frac_plot <- frac_plot + theme(axis.text.x = element_text(angle = 45, vjust = 0.9, hjust=1))

  plots <- list(rate=rate_plot, fraction=frac_plot)
  plots
}


compute_bootstrapped_data <- function(dat, R=1000, conf=0.95) {
    cutoffs <- unique(dat$cutoff)
    results <- list()
    for(cutoff in cutoffs) {
      reset_iter_count()
      print(paste(cutoff, "days"))
      dat_cutoff <- dat[dat$cutoff == cutoff, ]
      boot_out <- boot(dat_cutoff, compute_stats, R=R)
      boot_df <- assemble_ci_dataframe(boot_out, conf=conf)
      boot_df$cutoff <- cutoff
      results[[cutoff]] <-  boot_df
    }

    results_df <- do.call(rbind, results)
    results_df
}

main <- function(args) {
  theme_set(theme_grey(base_size = 18))
  raw_dat <- read.csv(args$input_path, stringsAsFactors=FALSE)
  raw_dat$ytrue <- ifelse(raw_dat$ytrue == "True", 1.0, 0.0)

  dat <- compute_bootstrapped_data(raw_dat, R=opt$R, conf=opt$conf)

  comparisons <- strsplit(args$comparisons, ":")[[1]]
  plots <- build_plots(dat, raw_dat, comparisons)

  csv_path <- paste0(args$output_dir, "/summary.csv")
  write.csv(dat, file=csv_path, row.names=FALSE)

  for(plot_name in names(plots)) {
    print(paste("Saving", plot_name, "plot"))
    plot_obj <- plots[[plot_name]]
    plot_path <- paste0(args$output_dir, "/", plot_name, ".pdf")
    ggsave(plot_path, plot_obj)
  }
}


option_list = list(
  make_option(c("--input_path"), type="character",
              help="Input file"),
  make_option(c("--R"), type="numeric", default=1000,
              help="Bootstrap replications"),
  make_option(c("--conf"), type="numeric", default=0.95, help="CI"),
  make_option(c("--comparisons"), type="character", default="Low-Intermediate:Intermediate-High", help="Comparisons"),
	make_option(c("--output_dir"), type="character",
              help="Path to directory for results")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)


if (is.null(opt$input_path) || is.null(opt$output_dir)) {
  print_help(opt_parser)
} else {
  main(opt)
}
