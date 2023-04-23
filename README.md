dev notes:

1. scripts to get LibriTTS are in data/get_data_scripts
2. to run an experiment run python run_test.py -c ./exp_configs/sound_reprod_lucid.yaml -n test.
   - c -config path
   - n - experiment / run name
4. exp_configs contains degub flag to test on dummy dataset

TODO:

- [ ] Add our loggin and wand db
- [ ] Remove model saving from trainer, move it to logger
- [ ] Add exp_name saving into model dict
- [ ] Check coding from SoundStream
- [ ] Remove installable ResidualVQ, add raw code files
- [ ] Train on a bigger dataset
- [ ] Compar results and metrics

# Personal Neural Codec

This project aims to build a machine learning model for encoding and decoding voice data. The model focuses on learning efficient representations of voice while maintaining high-quality audio reconstruction.

## Experiments

The following table summarizes the experiments conducted in this project:

| Experiment Name | Experiment Description                                               |
| --------------- | -------------------------------------------------------------------- |
| Experiment 1    | VQ-VAE model trained on mel spectrograms for audio encoding/decoding |
| Experiment 2    | TBD                                                                  |
| Experiment 3    | TBD                                                                  |
