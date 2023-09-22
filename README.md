# attacks-personalized-models-release
Code for investigating the effects of attacks on personalized ML models

# Setup
 - Download the ZIP file of the repository and extract.
 - `cd attacks-personalized-models-release/app/`
 - `pip install -r pip_requirements.txt`

# Experiments
## Pilot Experiments
 - aggression prediction task: `python -m app.experiments.z_pilots.aggression`
 - sentiment prediction task: `python -m app.experiments.z_pilots.sentiment`
## CP Experiments
 - aggression prediction task: `python -m app.experiments.wiki_detox_aggression.distilbert_CP`
 - sentiment prediction task: `python -m app.experiments.goemo_sentiment.distilbert_CP`
## MAL Experiments
 - aggression prediction task: `python -m app.experiments.wiki_detox_aggression.distilbert_MAL`
 - sentiment prediction task: `python -m app.experiments.goemo_sentiment.distilbert_MAL`

# Licenses
This repository contains two datasets that have been modified from their original versions:
 - Wikipedia Talk Labels: Aggression. Modification includes filtering out some texts and annotations, poisoning, and partitioning into training and test files. The original dataset was released under a CC0 public domain dedication. The original dataset can be found in:
 > Wulczyn, Ellery; Thain, Nithum; Dixon, Lucas (2017). Wikipedia Talk Labels: Aggression. figshare. Dataset. https://doi.org/10.6084/m9.figshare.4267550.v5
 - GoEmotions. Modification includes filtering out some texts and annotations, poisoning, and partitioning into training and test files. The original dataset was released under the CC BY 4.0 International license, which can be found here: https://creativecommons.org/licenses/by/4.0/legalcode. The original dataset can be found in:
 > D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, S. Ravi, GoEmotions: A dataset of fine-grained emotions, in: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Association for Computational Linguistics, Online, 2020, pp. 4040–4054. https://github.com/google-research/google-research/tree/master/goemotions

All source files in this repository are released under the GPL-3.0 license, the text of which can be found in the LICENSE file.
