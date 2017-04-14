TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

HEADERS += \
    constants.h \
    spline.h

#LIBS += -LC:/Armadillo/newblas -llibblas
#LIBS += -LC:/Armadillo/newblas -lliblapack

#INCLUDEPATH += C:/Armadillo/include
#DEPENDPATH += C:/Armadillo/include
INCLUDEPATH += C:\CPP_Libraries\boost\boost_1_63_0
DEPENDPATH += C:\CPP_Libraries\boost\boost_1_63_0
#INCLUDEPATH += C:\CPP_Libraries\alglib-3.10.0.cpp.gpl\cpp\src
#DEPENDPATH += C:\CPP_Libraries\alglib-3.10.0.cpp.gpl\cpp\src
