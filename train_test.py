import train_textBPN
import eval_textBPN
import os

def checkImagePath(path):
	if os.path.isfile(path):
		return True
	else:
		return False

def makeTrainingQuery(model, dataset, epochs = 2, learning_rate = 0.001):

	query = {}
	path = "model/"
	if model == 2:
		query['net'] = 'deformable_resnet50'
		query['scale'] = 1
		query['resume'] = path+'2/'
	elif model == 1:
		query['net'] = 'resnet18'
		query['scale'] = 4
		query['resume'] = path+'1/'

	if dataset == 1:
		query['exp_name'] = 'TotalText'
		query['max_epoch'] = epochs
		query['resume'] += 'TotalText.pth'
	elif dataset == 2:
		query['exp_name'] = 'Ctw1500'
		query['max_epoch'] = epochs
		query['resume'] += 'Ctw1500.pth'
	elif dataset == 3:
		query['exp_name'] = 'TD500'
		query['max_epoch'] = epochs
		query['resume'] += 'TD500.pth'

	query['batch_size'] = 12
	query['gpu'] = 0
	query['input_size'] = 640
	query['optim'] = 'Adam'
	query['lr'] = learning_rate
	query['num_workers'] = 30

	return query

def makeTestQuery(model, dataset):

	query = {}
	path = "model/"
	if model == 2:
		query['net'] = 'deformable_resnet50'
		query['scale'] = 1
		query['path'] = path+'2/'
	elif model == 1:
		query['net'] = 'resnet18'
		query['scale'] = 4
		query['path'] = path+'1/'

	if dataset == 1:
		query['exp_name'] = 'TotalText'
		#query['max_epoch'] = epochs
		query['path'] += 'TotalText.pth'
		query['test_size'] = (640,1024)
		query['dis_threshold'] = 0.325
		query['cls_threshold'] = 0.9

	elif dataset == 2:
		query['exp_name'] = 'Ctw1500'
		#query['max_epoch'] = epochs
		query['path'] += 'Ctw1500.pth'
		query['test_size'] = (640,1024)
		query['dis_threshold'] = 0.3
		query['cls_threshold'] = 0.925

	elif dataset == 3:
		query['exp_name'] = 'TD500'
		#query['max_epoch'] = epochs
		query['path'] += 'TD500.pth'
		query['test_size'] = (640,960)
		query['dis_threshold'] = 0.35
		query['cls_threshold'] = 0.9

	return query


## get user input and set values
print('Before proceeding, make sure you have copied all the datasets and models from google drive to the respective folders, have you copied ? yes or no')
choice = str(input())
if choice!='yes':
	exit()

while(True):
	print('Note: All models are trained to the respective performance metric. You can train them to some more epochs, test them on the original dataset or use one of the model to detect text on your supplied images')
	print('Menu: ')
	#print('1. Train a model\n2.Test a model\n3. Localize text on your image\n4.Exit\nEnter choice: ')
	print('1. Train a model\n2.Test a model\n4.Exit\nEnter choice: ')
	choice = int(input())
	if choice == 1:
		model = int(input('Choose model : \n1.Res18 + transformer\n2.Res50 + transformer\nEnter choice: '))
		dataset = int(input('Choose dataset : \n1.Total text\n2.CTW1500\n3.MSRA-TD500\nEnter choice: '))
		epochs = int(input('Enter how many epochs you want to train: '))
		learning_rate = int(input('Enter learning_rate: '))
		query = makeTrainingQuery(model, dataset,epochs,learning_rate)
		train_textBPN.main(query)

	elif choice == 2:
		model = int(input('Choose model : \n1.Res18 + transformer\n2.Res50 + transformer\nEnter choice: '))
		dataset = int(input('Choose dataset : \n1.Total text\n2.CTW1500\n3.MSRA-TD500\nEnter choice: '))
		query = makeTestQuery(model, dataset)
		eval_textBPN.main(query, 'viz')

	elif choice == 3:
		model = int(input('Choose model : \n1.Res18 + transformer\n2.Res50 + transformer\nEnter choice: '))
		dataset = int(input('Choose dataset : \n1.Total text\n2.CTW1500\n3.MSRA-TD500\nEnter choice: '))
		image_path = str(input("Enter full image path: "))
		if checkImagePath(image_path)==True:
			print("Image found .. processing started")

		else:
			print("Image not found")
		
	elif choice == 4:
		print('Program ends')
		exit()

	else:
		print('Wrong input')
