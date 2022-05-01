
__author__ = "Rubén Buzón Pérez"
__email__ = "ruben.buzon@live.u-tad.com"

import argparse
import utils

modes_classes = {
	'basic': ["Neutral", "Feliz", "Triste"],
	'extra': ["Neutral", "Feliz", "Triste", "Sorprendido", "Enfadado"],
}

modes_models = {
	'basic': './models/emotions_3_classes.hdf5',
	'extra': './models/emotions_5_classes.hdf5',
}

modes_classifiers = {
	'basic': './models/haarcascade.xml',
	'extra': './models/haarcascade.xml',
}




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Facial emotion recognition using neural network.')
	parser.add_argument('--class-mode', type=str, default='basic', choices=['basic', 'extra'], required=False,
						help="""Number of classes to detect. Basic mode classifies 3 diferent emotions with 87%%
						 of accuracy while Extra mode classifies 5 different classes with 74%% of accuracy.""")
	parser.add_argument('--input-mode', type=str, default='webcam', choices=['webcam', 'screen'], required=False,
						help="""Input source to detect faces(WIP).""") 
	# TODO: implementar grabación de pantalla
	parser.add_argument('--background', action=argparse.BooleanOptionalAction,
						help="""Execute in background without window.""")

	args = parser.parse_args()

	model = None
	model_classes = None
	classifier = None
	start_method = None

	match args.class_mode:

		case 'basic':
			model = modes_models.get('basic')
			model_classes = modes_classes.get('basic')
			classifier = modes_classifiers.get('basic')
			
		case 'extra':
			model = modes_models.get('extra')
			model_classes = modes_classes.get('extra')
			classifier = modes_classifiers.get('extra')
			
		case _:
			raise ValueError(f"Invalid argument {args.class_mode} for --class-mode")
	
	detector = utils.EmotionDetector(model=model, classifier=classifier, classes=model_classes, background=args.background)

	match args.input_mode:

		case 'webcam':
			start_method = detector.start_webcam_detection
			
		case 'screen':
			start_method = detector.start_screen_detection
			
		case _:
			raise ValueError(f"Invalid argument {args.input_mode} for --input-mode")

	start_method()


	