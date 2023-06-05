import blenderproc as bproc
import h5py
from matplotlib import pyplot as plt
import numpy as np



# f = h5py.File('examples/automate_semantic_relations/annotations/valE_raw.h5', 'r')

# print(f.keys())
# groups = list(f.keys())
# print(list(f.keys()))

# print(f['473']['attributes'][:,:])
# print(list(f.keys()))
# print(f['473'].keys())
# print(f['473']['attributes'])
# print(f['473']['attributes'][:,:])
# print(f['473']['bboxes'])
# print(f['473']['bboxes'][:,:])
# print(f['473']['image'])

# for i in range(int(groups[0]),int(groups[-1]) + 1):
    
#     if str(i) in groups:
#         # print(f[str(i)]['objects'])
#         # print(f[str(i)]['objects'][:])
#         # print(i)
#         # print(f[str(i)]['attributes'][:,:])
#         # print(f[str(i)]['relations'])
#         # print(f[str(i)]['relations'][:,:,:])
#         print(f[str(i)]['bboxes'])
#         print(f[str(i)]['bboxes'][:,:])

# """
# data = f[str(i)]['image'][:,:,:]

# plt.imshow(data, interpolation='nearest')
# # plt.show()
# plt.savefig("mygraph.png") """

# f.close()

# d = h5py.File('examples/automate_semantic_relations/Test_07/output/val.h5', 'w')

""" image_number = 0

for image_number in range(6):
    d.create_group(str(image_number))
    d[str(image_number)].create_dataset('attributes', (1, 2), dtype='f8')
    d[str(image_number)].create_dataset('bboxes', (2, 4), dtype='f8')
    d[str(image_number)].create_dataset('image', (480, 640, 3), dtype='|u1')
    d[str(image_number)].create_dataset('objects', (2,))
    d[str(image_number)].create_dataset('relations', (2, 2, 2), dtype='f8')

print(list(d.keys()))
print(list(d['1'].keys()))
print(d['1']['attributes'])
print(d['1']['bboxes'])
print(d['1']['image'])

print(d['1']['objects'])
print(d['1']['relations'])

number_elements = 2
number_relations = 2

relations = np.array(number_relations * [-1*np.eye(number_elements)])

print(relations) """


d = h5py.File('examples/automate_semantic_relations/Test_09/output/val.h5', 'r')

print(d["/"].keys())

groups = list(d["/"].keys())

print(d[groups[0]].keys())

for i in range(int(groups[0]),int(groups[-1]) + 1):

    if str(i) in groups:
        print("============================================================")
        print(i)
        print(d[str(i)]['image'])
        print(d[str(i)]['attributes'])
        print(d[str(i)]['attributes'][:,:])
        print(d[str(i)]['objects'])
        print(d[str(i)]['objects'][:])
        print(d[str(i)]['relations'])
        print(d[str(i)]['relations'][:,:,:])
        print(d[str(i)]['bboxes'])
        print(d[str(i)]['bboxes'][:,:])
        print(d[str(i)]['bboxes2'])
        print(d[str(i)]['bboxes2'][:,:])

d.close()