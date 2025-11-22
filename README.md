# Softmax Regression

An implementation of softmax regression used for classification.

![Status](https://img.shields.io/badge/status-beta-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
<!--[![Tests](https://img.shields.io/badge/tests-%20none-lightgrey)]()-->
<br>

---
<br>

## Overview

<!--
<p align="center">
  <img src="MENACE_matchboxes.png" alt="Original MENACE Matchboxes" width="400"/>
</p>
-->

Softmax Regression (also known as multinomial logistic regression) is a supervised learning algorithm used for multi-class classification. It generalizes binary logistic regression to cases where each input must be assigned to one of several mutually exclusive classes (e.g. digits, types of flowers, categories of images).

At its core, softmax regression models the probability that a sample input x belongs to class k by learning a separate weight vector for each class and then normalizing their outputs into a valid probability distribution.

## Project Status

### Planned
- Generalize to any CSV dataset.

### Known Limitations
- **Testing**: This project lacks automated unit tests. Edge cases may not be fully covered.
- **Compatibility**: This implementation is not yet general and has some things hard coded for the Iris dataset.
<br><br>

---
<br>

## Project Info
**Status:** Beta (usable, but lacks generalization)    
**Author:** T. Stratton  
**Start Date:** 1-NOV-2025  
**License:** MIT License – see [LICENSE](./LICENSE)  
**Language:** Python 3.11+ (tested on 3.11)   
**Topics:** softmax, regression, ai, supervised-learning, machine-learning