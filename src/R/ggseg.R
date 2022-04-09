library(ggsegDKT)
library(ggseg)
library(ggplot2)
library(tidyverse)
library(ggthemes)

df = read.csv("/Users/joshua/Developer/CognitiveSubtypes/notebooks/rois.csv") %>%
  pivot_longer(cols = c("Area", "Thickness", "Volume"), names_to = "measure")

cortical_pos <- c("left lateral", "left medial", "right medial", "right lateral")

df %>%
  group_by(measure) %>%
  ggplot() +
  ggtitle("Feature Importance in Random Forest Model") +
  theme_tufte() +
  geom_brain(
    atlas = dkt, aes(fill = value), 
    position = position_brain(cortical_pos), 
    show.legend = TRUE) +
  scale_fill_gradientn(
    name = "Mean Decrease in Impurity",
    colours = c("#CFE8F3", "#A2D4EC", "#73BFE2","#46ABDB", 
                "#1696D2", "#12719E", "#0A4C6A", "#062635")) +
  scale_x_continuous(breaks = c(140, 520, 900, 1280), 
                     labels = str_to_title(cortical_pos)) +
  theme(
    legend.position = 'bottom',
    plot.title = element_text(hjust = 0.5),
    legend.title = element_text(vjust = .75),
    axis.title.y=element_blank(), 
    axis.text.y=element_blank(), 
    axis.ticks.y=element_blank(),
    strip.text.y = element_text(angle = 0, hjust = 0)) +
  facet_grid(measure ~ .)

ggsave('/Users/joshua/Developer/CognitiveSubtypes/results/figures/fig5.jpg', dpi=300)
