"""
MalGAN -- GAN-Based Data Augmentation for Malware Image Classification.
=======================================================================

A pipeline that trains a Conditional DCGAN to generate synthetic malware
images, then uses those images to augment a ResNet50 classifier for
improved malware family recognition on the MaleVis dataset.
"""
