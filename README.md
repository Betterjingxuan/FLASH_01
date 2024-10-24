# FLASH: A Fast and Reliable Shapley Value Approximation Framework for Model-Agnostic Interpretation

This repository contains the code for the paper "FLASH: A Fast and Reliable Shapley Value Approximation Framework for Model-Agnostic Interpretation".
![Diagram of Shapley Value](figure/Introduction2.0.jpg "Shapley Value Approximation Diagram")

## FLASH Overview
FLASH is a framework supporting model-agnostic interpretation via fast and reliable Shapley value approximation. It employs a two-phase evaluation process: first, a layer-wise evaluation to generate unique coalitions in a pattern growth manner, followed by a feature-wise evaluation that focuses on the top-𝑘 features with the highest variances. FLASH also dynamically allocates the number of evaluations, ensuring a more efficient and reliable Shapley value approximation.
![Overview of FLASH](figure/OverviewMethod.jpg "Overview of FLASH")

## Features
- Model-agnostic 
- Local interpretation
- Dynamic Allocation

## Implementation
1. Prerequisites
- Java Development Kit (JDK) version 8 or higher
- An IDE (e.g., IntelliJ IDEA, Eclipse) or a text editor (e.g., VS Code)
  
2. Clone the Repository
   To get a local copy of this repository, run the following command:
    ```bash
    git clone https://anonymous.4open.science/r/FLASH-7088/

3. Compile and Run
- Navigate to the project directory:
  ```bash
   cd repository
- Compile the Java files
   ```java
   javac Main.java
- Run the application
  ```bash
  java Main

## Repository Structure
### Repository Root Directory
- Dataset: A directory for datasets used by this project.
- Model: A directory containing scripts to preprocess the dataset, train and deploy the model.
- src: The directory where the project's source code is stored.
- pom.xml: The Maven project configuration file that defines project dependencies, build settings, and other metadata.

### Folders
- src/main/java/AlgoVersion: A folder houseing all the algorithm-related code for this project. 
- src/main/java/Game: It contains the implementation of the game structure, utility functions and the mechanism to compute the Shapley value.
- src/main/java/Global: A directory for global variables and declarations that are shared across the project, providing a centralized place for common configurations and constants.
- src/main/java/Structure: A folder dedicated to the data structures used in FLASH. 

### Files
- main.java: All algorithms are implemented in 'main.java'
- Game/GameClass.java: The global initialization and experimental setup.
- Global/Info.java: All settings related to determinism.

- ## Baseline Setup
- MC: The Monte Carlo method (MC) approximates the Shapley value by randomly sampling permutations of features. In the experiments, MC serves as a benchmark for approximating the Shapley value.
- CC: It reformulates Shapley value estimation by using complementary contributions, measuring the utility difference between a coalition and its complement. <[CC-Method](https://github.com/ZJU-DIVER/ShapleyValueApproximation)>
- CCN: Building on the CC method, CCN optimizes the sampling process using Neyman allocation. <[CCN-Method](https://github.com/ZJU-DIVER/ShapleyValueApproximation)>
- S-SVARM: It is designed to approximate the Shapley value by sampling coalitions without relying on marginal contributions. <[S-SVARM method](https://github.com//kolpaczki//Approximating-the-Shapley-Value-without-Marginal-Contributions)>
