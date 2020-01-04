import numpy as np
import keras.backend as K
import cv2

class GradCam:
    def __init__(self, x, model, layername):
        self.x = x
        self.model = model
        self.layername = layername

    def get_classoutput(self):
        predictions = self.model.predict(self.x)
        class_idx = np.argmax(predictions[0])
        class_output = self.model.output[:, class_idx]
        return class_output

    def calc_cam(self):
        class_output = self.get_classoutput()
        conv_output = self.model.get_layer(self.layername).output
        grads = K.gradients(class_output, conv_output)[0]
        grad_function = K.function([self.model.input], [conv_output, grads])

        output, grads_val = grad_function([self.x])
        output, grads_val = output[0], grads_val[0]

        weights = np.mean(grads_val, axis=(0,1))
        cam = np.dot(output, weights)
        return cam
    
    def integrate(self):
        cam = self.calc_cam()
        cam = cv2.resize(cam, self.x.shape[1:3], cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        temp_x = self.x[0,:,:,0]
        temp_x = temp_x / temp_x.max()

        jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET) 
        jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB) 
        jetx = cv2.applyColorMap(np.uint8(255 * temp_x), cv2.COLORMAP_JET)
        jetx = cv2.cvtColor(jetx, cv2.COLOR_BGR2RGB)
        jetcam = (np.float32(jetcam) + jetx) / 2
        return cam, jetcam

if __name__=='__main__':
    import numpy as np

    temp = np.arange(25).reshape(5,5)
    temp = np.mean(temp, axis=(0,1))
    print(temp)