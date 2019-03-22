import os
root_dir = "/home/songanz/Documents/Git_repo/fusion/"
label_dir = root_dir + "data/KITTI/training/label_2/"
label_files = os.listdir(label_dir)
label_files.sort()

# refer to kitti.name
DontCare = 0
Car = 0
Van = 0
Truck = 0
Pedestrian = 0
Person_sitting = 0
Cyclist = 0
Tram = 0
Misc = 0

nothing = 0
check = 0

for index in range(len(label_files)):

    label = open(label_dir + label_files[index], 'r')
    label_contents = label.readlines()

    for line in label_contents:

        data = line.split(' ')
        if len(data) == 15:
            class_str = data[0]
            if class_str == 'DontCare':
                DontCare += 1
            elif class_str == 'Car':
                Car += 1
            elif class_str == 'Van':
                Van += 1
            elif class_str == 'Truck':
                Truck += 1
            elif class_str == 'Pedestrian':
                Pedestrian += 1
            elif class_str == 'Person_sitting':
                Person_sitting += 1
            elif class_str == 'Cyclist':
                Cyclist += 1
            elif class_str == 'Tram':
                Tram += 1
            elif class_str == 'Misc':
                Misc += 1
            else:
                nothing += 1
        else:
            check += 1
            print("Label index: ", index, "  Line: ", line)

print("************Result****************")
print("DontCare: ", DontCare, "\n")
print("Car: ", Car, "\n")
print("Van: ", Van, "\n")
print("Truck: ", Truck, "\n")
print("Pedestrian: ", Pedestrian, "\n")
print("Person_sitting: ", Person_sitting, "\n")
print("Cyclist: ", Cyclist, "\n")
print("Tram: ", Tram, "\n")
print("Misc: ", Misc, "\n")
print("nothing: ", nothing, "\n")
print("check: ", check, "\n")
