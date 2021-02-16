classification_metrics.npz contains LOOCV results from marr using SVMs 
and the following features:

'PWR', 'PWR_3SEC', 'PWR_9SEC', 'PWR_27SEC', 'VLTG'

Best stim-conservative results are obtained with C=.064 Best 
stim-lenient results are obtained with C=.052 (which gives best valid balanced 
accuracy of ~0.65, valid sens=0.9, valid spec=0.4)

train_modular.py was used to get this
