# Holistic<sup>++</sup> Scene Understanding

This repo contains code for ICCV 2019 paper.

[Holistic++ Scene Understanding: Single-view 3D Holistic Scene Parsing and Human Pose Estimation with Human-Object Interaction and Physical Commensense](https://yixchen.github.io/holisticpp/file/holistic_scenehuman.pdf)  
Yixin Chen*, Siyuan Huang*, Tao Yuan, Siyuan Qi, Yixin Zhu, Song-Chun Zhu  
*The IEEE International Conference on Computer Vision (ICCV)*, 2019   
(* indicates equal contribution.)

In this paper, we propose a new 3D holistic<sup>++</sup> scene understanding
problem, which jointly tackles two tasks from a single-view
image: (i) holistic scene parsing and reconstruction—3D estimations
of object bounding boxes, camera pose, and room
layout, and (ii) 3D human pose estimation. The intuition behind
is to leverage the coupled nature of these two tasks to
improve the granularity and performance of scene understanding.
We propose to exploit two critical and essential
connections between these two tasks: (i) human-object interaction
(HOI) to model the fine-grained relations between
agents and objects in the scene, and (ii) physical commonsense
to model the physical plausibility of the reconstructed
scene. The optimal configuration of the 3D scene, represented
by a parse graph, is inferred using Markov chain
Monte Carlo (MCMC), which efficiently traverses through
the non-differentiable joint solution space. Experimental results
demonstrate that the proposed algorithm significantly
improves the performance of the two tasks on three datasets,
showing an improved generalization ability.

Please refer to our <a href="https://yixchen.github.io/holisticpp/file/holistic_scenehuman.pdf"> paper </a> or <a href="https://yixchen.github.io/holisticpp/">project </a> for more details.

# Usage
## Install
Support Python 3+.
```
pip install -r requirements.txt
```
## Data
1. Download the preprocessed data which contains the image and initialization from <a href="https://drive.google.com/file/d/1RBsGUSFze0z49iGTo2YBe_mx8P0h8S8I/view?usp=sharing"> here </a>, extract it with

        tar -vzxf data.tar.gz       

We will release more images from different dataset with initialization soon.
## Inference

Joint inference of holistic scene understanding and human pose by image name. 
    
        python inference_natural_image.py --image_name 1


## Citation

If you find the paper and/or the code helpful, please cite us.

```
@inProceedings{chen2019holisticpp, 
title={Holistic++ Scene Understanding: Single-view 3D Holistic Scene Parsing and Human Pose Estimation with Human-Object Interaction and Physical Commonsense}, 
author = {Chen, Yixin and Huang, Siyuan and Yuan, Tao and Qi, Siyuan and Zhu, Yixin and Zhu, Song-Chun}, 
booktitle={The IEEE International Conference on Computer Vision (ICCV), 
year={2019} 
}
```
## License

Our code is released under the MIT license.
        