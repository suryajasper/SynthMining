cin = open('list_attr_celeba.txt', 'r')

lines = [line.strip().split(' ') for line in cin.readlines()]
categories = lines[0]

cout = open('formatted_attrs.txt', 'w')

for img_line in lines[1:]:
  img_line = [attr for attr in img_line if attr != '']
  fname = img_line[0]

  img_cats = [
    categories[i] 
      for i, val in enumerate(img_line[1:]) 
        if int(val) == 1
  ]

  cout.write(f'{fname} {",".join(img_cats)}\n')

cin.close()
cout.close()