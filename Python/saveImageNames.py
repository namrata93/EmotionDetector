import os


target = open('names.txt' , 'w')
for filename in os.listdir('/Users/namrataprabhu/Documents/mmi/UpdatedImages/'):
	imageName = '/Users/namrataprabhu/Documents/mmi/UpdatedImages/' + filename
	target.write(imageName)
target.close()
