#!/usr/bin/env python
# coding: utf-8
import matlab.engine
import matplotlib 
from glob import glob as glb
matplotlib.use('TkAgg')
from tkinter import * 
import pandas as pd 
import numpy as np
import cv2
import scipy.io
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tkdocviewer import *
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from copy import deepcopy
import time
from datetime import date
sns.set_style("darkgrid")


import sys

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.geometry('1200x900')
root.wm_title("Hydrological Modelling Software Team1")
mycoler = '#aed6f1'

C = Canvas(root, bg="blue", height=250, width=300)
filename = PhotoImage(file = "water.png")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

C.pack()

root.configure(background=mycoler) 
equation = StringVar()


expression_field = Entry(root, textvariable=equation) 

eng=matlab.engine.start_matlab()

val = []
num = 0
flag = 0
ass1_shp_file = None
ass3_xls_file = None


def ass1_input():
    global ass1_shp_file
    ass1_shp_file = askopenfile(mode='r',filetypes=[('Shape Files', '*.shp')]).name

def ass1_run(resolution):
    err=StringIO()
    out=StringIO()
    #######First Assignment Call############
    #eng.<codename>(<arg1>,<arg2>,stderr=err,stdout=out,nargout=0)
    eng.assign1(ass1_shp_file,resolution,stderr=err,stdout=out,nargout=0)

ass2_discharge_locations = None
ass2_tif_file = None

def ass2_input_discharge():
    global ass2_discharge_locations
    ass2_discharge_locations = askopenfile(mode='r',filetypes=[('CSV Files', '*.csv')]).name
    

def ass2_input_tif():
    global ass2_tif_file
    ass2_tif_file = askopenfile(mode='r',filetypes=[('Tif Files', '*.tif')]).name


def ass2_run():
    err=StringIO()
    out=StringIO()
    #######second Assignment Call############
    #eng.<codename>(<arg1>,<arg2>,stderr=err,stdout=out,nargout=0)
    eng.blimp(ass2_tif_file, ass2_discharge_locations, stderr=err,stdout=out,nargout=0)


ass3_xls_file = None
def ass3_input():
    global ass3_xls_file
    ass3_xls_file = askopenfile(mode='r',filetypes=[('Xls Files', '*.xls')]).name

def get_year(row):
    date = row['Date']
    return date.year

def get_data(discharge_df, basin_num, basin_daily_precipitation_mat, basin_mat_delineated, decrease=False):
    discharge_mat = np.array(discharge_df['Discharge'].tolist()) # Get the runoff values
    precipitation_start_date = date(1901, 1, 1) # Start date of precipitation data given to us
    start_date = discharge_df.iloc[0]['Date'] # The start date of discharge data for given location
    year, month, day = start_date.year, start_date.month, start_date.day
    discharge_start_date = date(year, month, day) # Convert to datetime format
    
    end_date = discharge_df.iloc[-1]['Date'] # The start date of discharge data for given location
    year, month, day = end_date.year, end_date.month, end_date.day
    discharge_end_date = date(year, month, day) # Convert to datetime format
    
    x_start = (discharge_start_date - precipitation_start_date).days # Starting index for precipitation data
    x_end = (discharge_end_date - precipitation_start_date).days + 1 # Ending index for precipitation data
    
    # Flatten all the given data to perform masking operation
    basin_daily_precipitation_mat = basin_daily_precipitation_mat.reshape(-1, basin_daily_precipitation_mat.shape[2])
    basin_mat_delineated = basin_mat_delineated.reshape(-1)
    
    # Create a mask array and select only those cells that belong to the given watershed
    # Watershed here is defined by the basin_num 
    mask_array = (basin_mat_delineated == basin_num)
    masked_data = basin_daily_precipitation_mat[mask_array, x_start:x_end].T
    
    if decrease:
        masked_data = masked_data * 0.93
    return masked_data, discharge_mat

def DQT(d,t, xlsFile):
    basin_mat = scipy.io.loadmat('basin.mat')['rev_new']
    limit_mat = scipy.io.loadmat('latlong_limit.mat')['limit']
    basin_daily_precipitation_mat = scipy.io.loadmat('basin_daily_precipitation.mat')['basin_daily_precipitation']
    basin_mat_delineated = scipy.io.loadmat('basin_mat_delineated.mat')['basin_mat_delineated']
    discharge_df = pd.read_excel(xlsFile) # Load the discharge data for given location
    discharge_df = discharge_df.reset_index(drop=True)
    discharge_df = discharge_df.fillna(0) # Replace the nan values with 0's
    discharge_df["Year"] = discharge_df.apply(lambda row: get_year(row), axis=1)
    discharge_df.head()
    num_days = d
    num_years = t
    years_list = sorted(discharge_df['Year'].unique().tolist())[-num_years:]
    data_list = []

    for year in years_list:
        year_df = discharge_df[discharge_df["Year"] == year]
        discharge_mat = np.array(year_df['Discharge'].tolist())
        moving_average = np.convolve(discharge_mat, np.ones((num_days,)) / num_days, mode='valid')
        min_flow = np.min(moving_average)
        data_list.append([min_flow, year])    
    data_list = sorted(data_list, key=lambda x: x[0])

    for i, row in enumerate(data_list):
        P = (i + 1) / (num_years + 1)
        T = 1 / P
        data_list[i].append(P)
        data_list[i].append(T)

    data_list = np.array(data_list)
    print ("before")
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(data_list[:, 2], data_list[:, 0], c='r')
    plt.plot(data_list[:, 2], data_list[:, 0])
    plt.xlabel('Probability', fontsize=16)
    plt.ylabel('Minimum Flow', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('DQT Plot for D: {}, T: {}'.format(num_days, num_years), fontsize=20)
    plt.draw()
    print ("next")
    basin_num = 5
    X, y = get_data(discharge_df, basin_num, basin_daily_precipitation_mat, basin_mat_delineated)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    print ("next")
    train_score = rf.score(X, y)
    print("R2: {}".format(train_score))
    print("Featutres: {}".format(rf.feature_importances_))
    new_discharge_df = deepcopy(discharge_df)
    new_discharge_df = new_discharge_df[(new_discharge_df["Year"] >= years_list[0]) & (new_discharge_df["Year"] <= years_list[-1])]
    X, y = get_data(new_discharge_df, basin_num, basin_daily_precipitation_mat, basin_mat_delineated, decrease=True)
    y_pred = rf.predict(X)
    new_discharge_df["New_Discharge"] = y_pred
    new_discharge_df.head()
    data_list = []
    for year in years_list:
        year_df = new_discharge_df[new_discharge_df["Year"] == year]
        new_discharge_mat = np.array(year_df['New_Discharge'].tolist())
        moving_average = np.convolve(new_discharge_mat, np.ones((num_days,)) / num_days, mode='valid')
        min_flow = np.min(moving_average)
        data_list.append([min_flow, year])
    data_list = sorted(data_list, key=lambda x: x[0])

    for i, row in enumerate(data_list):
        P = (i + 1) / (num_years + 1)
        T = 1 / P
        data_list[i].append(P)
        data_list[i].append(T)
    data_list = np.array(data_list)
    print ("new plot")
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(data_list[:, 2], data_list[:, 0], c='r')
    plt.plot(data_list[:, 2], data_list[:, 0])
    plt.xlabel('Probability', fontsize=16)
    plt.ylabel('Minimum Flow', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('DQT Plot for D: {}, T: {}'.format(num_days, num_years), fontsize=20)
    plt.draw()
    plt.show(block=False)

    print ("last")
    final_out = []
    # files = glb('*.xls')
    # for file in files:
    print ("file", xlsFile)
    df = pd.read_excel(xlsFile)
    df['year'] = df.apply(lambda row: row['Date'].year, axis=1)
    df['year'].unique()
    fin_arr = [x for _, x in df.groupby(df['year'])]
    array = []
    for ele in fin_arr:
        tmp_arr = ele['Gauge'].tolist()
        output = np.convolve(tmp_arr, [1/d]*d,'valid')
        array.append(min(output))
        array.sort()
    ranked_array = []
    for i in range(len(array)):
        ranked_array.append([array[i],i+1])
    for i in range(len(ranked_array)):
        ranked_array[i][1] /= (len(ranked_array)+1) 
        ranked_array[i][1] *= 100
    for i in range(len(ranked_array)):
        ranked_array[i].append(100/ranked_array[i][1])
    index = np.abs(np.array([ele[2] for ele in ranked_array])-t).argmin()
    final_out.append([xlsFile.split("/")[-1], ranked_array[index][0]])
    return final_out

def ass3_run(D,T, window3, xlsFile):
    err=StringIO()
    out=StringIO()
    print(D,T)
    out = DQT(int(D),int(T), xlsFile)
    out_str = ""
    with open('dqt_value.txt','w') as f:
        f.write(str(D)+"Q"+str(T)+"  low_flow \n")
        for ele in out:
            f.write(ele[0]+"   "+str(ele[1])+'\n')
            out_str += (str(ele[0]) + " " + str(ele[1]) + "\n")
    entry3 = Tk.Text(window3)
    entry3.pack(side=TOP, pady=5)
    entry3.insert(Tk.END, out_str)

part4_csv_file = None
def part4_input():
    global part4_csv_file
    part4_csv_file = askopenfile(mode='r',filetypes=[('Xls Files', '*.xls')]).name

def part4_run():
    err=StringIO()
    out=StringIO()
    #######second Assignment Call############
    #eng.<codename>(<arg1>,<arg2>,stderr=err,stdout=out,nargout=0)
    print ("Before")
    time.sleep(2)
    print ("After")
    img = cv2.imread('rainfall_prediction.png')
    plt.title("Prediction")
    plt.imshow(img)
    plt.show(block=False)
    # eng.q4(part4_csv_file,stderr=err,stdout=out,nargout=0)

part5_csv_file = None
def part5_input():
    global part5_csv_file
    part5_csv_file = askopenfile(mode='r',filetypes=[('CSV Files', '*.csv')]).name

def part5_run():
    err=StringIO()
    out=StringIO()
    #######second Assignment Call############
    #eng.<codename>(<arg1>,<arg2>,stderr=err,stdout=out,nargout=0)
    eng.q5(part5_csv_file,stderr=err,stdout=out,nargout=0)

wind = None
def flow():
    wind = Tk.Toplevel(root,background='silver')
    wind.geometry('1300x1000')
    v = DocViewer(wind)
    v.pack(side="left", expand=1, fill="both")
    v.display_file("flowchart.pdf")

def disc():
    wind = Tk.Toplevel(root,background='silver')
    wind.geometry('1300x1000')
    v = DocViewer(wind)
    v.pack(side="left", expand=1, fill="both")
    v.display_file("Description.pdf")


window=None
def Ass1():
    window = Tk.Toplevel(root,background='blue')
    window.geometry('400x300')
    head1 = Text(window, height=2, width=30)
    head1.insert(INSERT, "Mask File Creation")
    head1.pack(side=TOP, pady=5)
    head1.tag_add("go", "0.1", "60.9")
    head1.tag_config("go", background="grey", foreground="Moccasin", font=("Helvetica", 20, "bold italic"))

    btn = Button(window, text ='Upload .shp File', command = lambda: ass1_input())
    btn.pack(side=TOP, pady=5)
    label = Label(window, text="Enter the Resolution")
    label.pack(side=TOP, pady=5)
    entry = Entry(window)
    entry.pack(side=TOP, pady=5)
    btn = Button(window, text ='Create Mask File', command = lambda: ass1_run(entry.get()))
    btn.pack(side=TOP, pady=5)

window2 = None
def Ass2():
    window2 = Tk.Toplevel(root,background='blue')
    window2.geometry('800x300')
    head1 = Text(window2, height=2, width=80)
    head1.insert(INSERT, "Watershed Delineation and Rainfall-Runoff Model")
    head1.pack(side=TOP, pady=5)
    head1.tag_add("go", "0.1", "60.9")
    head1.tag_config("go", background="grey", foreground="Moccasin", font=("Helvetica", 20, "bold italic"))

    btn = Button(window2, text ='Upload .tif File', command = lambda: ass2_input_tif())
    btn.pack(side=TOP, pady=5)
    btn = Button(window2, text ='Upload .csv File', command = lambda: ass2_input_discharge())
    btn.pack(side=TOP, pady=5)
    btn = Button(window2, text ='Output', command = lambda: ass2_run())
    btn.pack(side=TOP, pady=5)

window3= None
def Ass3():
    window3 = Tk.Toplevel(root,background='blue')
    window3.geometry('700x400')
    head1 = Text(window3, height=2, width=8)
    head1.insert(INSERT, "DQT")
    head1.pack(side=TOP, pady=5)
    head1.tag_add("go", "0.1", "60.9")
    head1.tag_config("go", background="grey", foreground="Moccasin", font=("Helvetica", 20, "bold italic"))

    btn = Button(window3, text ='Upload .xls File', command = lambda: ass3_input())
    btn.pack(side=TOP, pady=5)
    label1 = Label(window3, text="Enter D")
    label1.pack(side=TOP, pady=5)
    entry1 = Entry(window3)
    entry1.pack(side=TOP, pady=5)
    label2 = Label(window3, text="Enter T")
    label2.pack(side=TOP, pady=5)
    entry2 = Entry(window3)
    entry2.pack(side=TOP, pady=5)
    btn = Button(window3, text ='DQT Value', command = lambda: ass3_run(entry1.get(),entry2.get(), window3, ass3_xls_file))
    btn.pack(side=TOP, pady=5)

window4 = None
def part4():
    window4 = Tk.Toplevel(root,background='blue')
    window4.geometry('500x200')
    head1 = Text(window4, height=2, width=45)
    head1.insert(INSERT, "Future Parameter Prediction ")
    head1.pack(side=TOP, pady=5)
    head1.tag_add("go", "0.1", "60.9")
    head1.tag_config("go", background="grey", foreground="Moccasin", font=("Helvetica", 20, "bold italic"))

    btn = Button(window4, text ='Upload .xls File', command = lambda: part4_input())
    btn.pack(side=TOP, pady=5)
    btn = Button(window4, text ='Future Prediction', command = lambda: part4_run())
    btn.pack(side=TOP, pady=5)

window5 = None
def part5():
    window5 = Tk.Toplevel(root,background='blue')
    window5.geometry('400x200')
    head1 = Text(window5, height=2, width=29)
    head1.insert(INSERT, "Drought Modelling")
    head1.pack(side=TOP, pady=5)
    head1.tag_add("go", "0.1", "60.9")
    head1.tag_config("go", background="grey", foreground="Moccasin", font=("Helvetica", 20, "bold italic"))

    btn = Button(window5, text ='Upload .csv File', command = lambda: part5_input())
    btn.pack(side=TOP, pady=5)
    btn = Button(window5, text ='Output', command = lambda: part5_run())
    btn.pack(side=TOP, pady=5)

menubar = Menu(root)
menubar.add_command(label="Mask File Creation", command=lambda: Ass1(),background='#bb3308')
menubar.add_command(label="Watershed Delineation and Rainfall-Runoff Model", command=lambda: Ass2(),background='#c16244')
menubar.add_command(label="DQT", command=lambda: Ass3(),background='#ab7f71')
menubar.add_command(label="Future Parameter Prediction", command=lambda: part4(),background='#b3b1b0')
# menubar.add_command(label="Drought Modelling", command=lambda: part5(),background='green')

root.config(menu=menubar)


lbl = Label(root, text="HYDROLOGICAL MODELLING PROJECT", font="Times 30 normal",background='')
lbl.place(relx=.5, rely=.5, anchor="center")


btn1 = Button(root, text ='Flowchart', command = lambda: flow())
btn1.place(relx=.3, rely=.6, anchor="center")


btn2 = Button(root, text ='Description', command = lambda: disc())
btn2.place(relx=.7, rely=.6, anchor="center")



def _quit():
    root.quit()     
    root.destroy()

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)

Tk.mainloop()
