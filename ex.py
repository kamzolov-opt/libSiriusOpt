import libname

print(libname.function())
print(libname.method.selffunction())
print(libname.module.function())
print(libname.module.method.function())
# print(module.selffunction()) Ошибка, selffunction не статично!
