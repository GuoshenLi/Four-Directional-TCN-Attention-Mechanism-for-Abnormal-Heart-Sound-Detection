# Four-Directional-TCN-Attention-Mechanism-for-Abnormal-Heart-Sound-Detection


## This is the model that I originally proposed during my undergraduate graduation theis 'Abnormal heart sound detection based on deep learning'.
## With this model, my thesis was graded excellent and I was awarded the Graduation (Thesis) Innovation Award (3%).

### Overview:

* The existing methods for detecting abnormal heart sound are easily interfered by noise and cannot extract the weak pathological features well. In view of the above problems, we creatively propose a feature map mask learning strategy based on four-directional temporal convolutional network, so as to fully capture the potential time-frequency correlation information of heart sound signal, and use it to design a convolutional neural network with attention mechanism.

* The backbone we use is the residual attention network, in which they use the traditional down sampling and upsampling strategy to construct the masking branch. Instead we use the four-directional TCN to do that.

### The intuition behind the 4-D TCN are the following:

* The image that we feed in the neural network is the mel spectrogram with log scale for the original heart sound, which is time related, so as the feature map in the neural network. In order to fully capture the time related information in the feature map, we use four TCN that sweeps horizontally and vertically in both directions across the feature map to fully capture the time-frequency correlation information of heart sound signal. What's more, the horizontally sweeps can enable the natwork to learn the time related information, while the vertically sweeps can help the network learn the information of energy of different frequency bands. The 4D-TCN model will create four mask, corresponding to the up, down, left, right direction. 

* Actually, you can plug in this softmask branch in any CNN architecture you like!


### The model architecture

<img src = https://github.com/GuoshenLi/Four-Directional-TCN-Attention-Mechanism-for-Abnormal-Heart-Sound-Detection/blob/main/tcn_masking_model.png><br/>

### The 4D-TCN atchitecture
<img src = https://github.com/GuoshenLi/Four-Directional-TCN-Attention-Mechanism-for-Abnormal-Heart-Sound-Detection/blob/main/4dtcn.png width = '418' height = '368'/><br/>

### The parameter of the network architecture
For more details, please refer to the code.



<img src = https://github.com/GuoshenLi/Four-Directional-TCN-Attention-Mechanism-for-Abnormal-Heart-Sound-Detection/blob/main/param1.png width = '662' height = '441'><br/>

<img src = https://github.com/GuoshenLi/Four-Directional-TCN-Attention-Mechanism-for-Abnormal-Heart-Sound-Detection/blob/main/param2.png width = '662' height = '441'/><br/>


### How do we scan the feature map in four direction? (A quick look)

```
The size of the feature map: Bs, H, W, C
We use the patches of (2, 2) and can reshape the tensor into Bs, #H, psh, #W, psw, C
Then, we transpose the tensor: Bs, #H, # W, psh, psw, C   (psh, psw == 2, 2)
So the information of each path (H, W) is in (psh, psw, C)
Therefore, we can further reshape the tensor into Bs * #H, #W, psh * psw *C
This is a 3D tensor and therefore can feed into a TCN. (left scan)
If we reverse the information in the #W direction, we can get the right scan.

```


### The model is implemented by tensorflow 1.x.


