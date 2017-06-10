from __future__ import print_function
import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os.path
import csv

parser = argparse.ArgumentParser(description=
                '''plot results''')
parser.add_argument('directory', help='directory help')
args = parser.parse_args()
directory = args.directory
print("plotting directory")

val_acc_file = os.path.join(directory, "val_accs.csv")
val_ce_file = os.path.join(directory, "val_ce.csv")
acc_file = os.path.join(directory, "accs.csv")
ce_file = os.path.join(directory, "ce.csv")

def read_files(filename):
  reader = csv.reader(filename)
  for row in reader:
    data_for_plotting

data_for_plotting = numpy.zeros((20, 4))



i = 0
for alpha in alpha_range:
    for p in range(len(m1_solution)):
        K.set_value(nn.model.trainable_weights[p], 
                    m2_solution[p]*alpha + m1_solution[p]*(1-alpha))
    train_xent, train_acc = nn.model.evaluate(X_train, y_train,
                                           batch_size=1024, verbose=0)
    test_xent, test_acc = nn.model.evaluate(X_test, y_test,
                                         batch_size=1024, verbose=0)
    data_for_plotting[i, :] = [train_xent, train_acc, test_xent, test_acc]
    i += 1

# finally, let's plot the data
# we plot the XENT loss on the left Y-axis
# and accuracy on the right Y-axis
# if you don't have Matplotlib, simply print
# data_for_plotting to file and use a different plotter

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(alpha_range, data_for_plotting[:, 0], 'b-')
ax1.plot(alpha_range, data_for_plotting[:, 2], 'b--')

ax2.plot(alpha_range, data_for_plotting[:, 1]*100., 'r-')
ax2.plot(alpha_range, data_for_plotting[:, 3]*100., 'r--')

ax1.set_xlabel('alpha')
ax1.set_ylabel('Cross Entropy', color='b')
ax2.set_ylabel('Accuracy', color='r')
ax1.legend(('Train', 'Test'), loc=0)

ax1.grid(b=True, which='both')
fn = os.path.basename(weight_file1)+'_' +os.path.basename(weight_file2) +'.pdf'
plt.savefig('Figures/'+ fn)
print('Plot save as ' + fn + ' in the Figures/ folder')
