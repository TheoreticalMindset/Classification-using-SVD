

import numpy as np # for fast calculations and to import images
import matplotlib.pyplot as plt # used to plot nice images of the digits
from numpy.linalg import svd # used to compute residuals and svds

######################################## REPLACE WITH YOUR PATH
TrainDigits = np.load('your path here')
TrainLabels = np.load('your path here')

TestDigits = np.load('your path here')
TestLabels = np.load('your path here')
###########################################


k = 10 # approximation rank (seach for low rank approximation if you're not familiar with this).
        # It basically how much information you want in your predictions. Higher is
        # better to some extent. Up to around 10-15 in this case.
train_images = 500 # number of training images 
test_images = 100 # number of test images

def import_images():
    """
    Import all the training images in 10 matrices with each column 
    representing an image with 784 pixel values.
    """

    matrices = [np.empty((784, 0))]*10 # 10 matrices, one for each digit
    for i in range(train_images): 
        image = np.row_stack(TrainDigits[:,i]) # stores each pixel the image as a column vector
        digit = TrainLabels[:,i][0] # gets the label of the digit

        if matrices[digit].shape[1] < 50: # only use 50 training images per digit
            matrices[digit] = np.column_stack([matrices[digit], image]) # insert column vector to training matrix

    return matrices # retutrns a matrix with a 10 matrices, one for each digit. 0-10.


def SVD(matrices):
    """
    This functions computes the SVD to each training matrix.
    """

    svds = [] # define list to store all matrices of the svds
    for matrix in matrices: # loop through all 10 matrices
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False) # returns the svd matrices
        svds.append([U, S, Vt]) # store the 3 matrices as a list in "svds" list

    return svds # returns the svd matrices
    

def setup(matrices, svds, k):
    """
    This functions sets up the program to solve the least square problems (one for each test digit).
    That is: to classify a digit we have to solve Ax=b. Or... not quite. It is enough to calculate the error this
    produces. That is, it's enough to calculate the residual to each test digit and then choosing the digit
    with the smallest relative error. It can be proven that the residual is I - U_k*U_k^t where I is the 
    identity matrix and U is a factor in our SVD. U_k represents the low rank approximation to U. So we
    basically have to compute I - U_k*U_k^T. The reason we use the low rank approximation is to fasten
    up the comnputation.

    The general approach to the problem is to do a linear regression to fit all training images. Then
    calculating the residual for a new test image. Then choosing the digit with the lowest residual.
    """

    Aks = [] # list to store the matrix we later use to compute the residuals/errors
    for index,matrix in enumerate(matrices):

        U = svds[index][0] # get the U matrix in the svd
        S = svds[index][1] # get the sigma matrix in the svd. that is the diagonal matrix with all the 
                           #singular values on the diagonal


     ######################################################## this part is not very important. 
                                                               #Its more to ensure stability
        eps = 2.2*10**-16 # machine error in the computer
        norm = np.linalg.norm(matrix, np.inf) # calculate the norm of the training matrix
        r = np.size(np.where(S > eps*norm)) # compute the numerical rank. To ensure we know the rank 
                                            # for the low rank appoximation
        U1 = U[:,:r] 
    ########################################################

        U_k = U1[:, :k] # lower the rank for U to fasten up the computations
        A = U_k @ U_k.T # calculate U_k * U_k^T
        rows, cols = A.shape
        I = np.eye(rows, cols) # create identity matrix to match the shape of A

        Ak = I-A

        Aks.append(Ak)

    return Aks


matrices = import_images() # a list with every matrix representing all training images for a digit
svds = SVD(matrices) #calculatesd svds for every matrix
Aks = setup(matrices, svds, k) # Aps is list with psuedo inverses and Ak is k:th approximation of A


def solve(Aks, b):
    """
    This function calculates the residual based on the information we got from the set up function.
    """

    res = []    
    for index in range(10): # the index gives which digit we are going to calculate the residual relative to
        res.append(np.linalg.norm(Aks[index] @ b)) # calculates the residual/error and stores it in a list

    min_res, min_index = min((value, index) for index, value in enumerate(res)) # returns the smallest residual 
                                                                                # and its index
    return min_index # only return the index. In this case the index is the same as the digit! e.g. 
                     # index 8 = digit 8 has the smallest residual!


############################################################### TEST LOOP
while True:
    dig=int(input("Digit: ")) # choose an image between 0 and the number of test images you chose
    TestLabel = TestLabels[:,dig][0] # get the test label

    b = np.row_stack(TestDigits[:,dig]) # store the test image as vector with the pixel values
    b_img = np.reshape(b, (28, 28)).T # Reshaping a vector to a matrix to plot nicely

    pre = solve(Aks, b) # predictions is given by the solve function

    print("Prediction:", pre, "Actual:", TestLabel)

    plt.imshow(np.reshape(b_img, (28, 28)), cmap ='Grays') # plot test image
    plt.show() 
#################################################################### you're done
