# cGAN-OSR
### Open-Set Recognition using Conditional GANs
![cGAN-OSR](draw/cGAN-OSR.jpg?raw=true "cGAN-OSR")


#### TO-DO Experiments
##### Stage 1
- [ ] Without Autoencoder Training
- [ ] With Autoencoder Training, to ensure encoder captures all information required for reconstruction
- [ ] With Variational Autoencoder Training
    - [ ] Separate Decoder, Generator 
    - [ ] Decoder == Generator

##### Stage 2
- [ ] Approach 1a: Random Noise, Expected Fake
- [ ] Approach 1b: Inverted Image, Expected Fake
- [ ] Approach 2: Random Image belonging to class c<sub>2</sub>, Expected c<sub>2</sub>
- [ ] Approach 3a: Random Noise, Expected Mismatch
- [ ] Approach 3b: Inverted Image, Expected Mismatch
- [ ] Approach 4: Inverted Image, Expected Inverted
