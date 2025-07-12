import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

#s = wb.series.info(q="")
#print(s)

#data = wb.series.list(q="inflation")

#for id in data:
#    print(id)

#To use the built-in plotting method
#wb.data.DataFrame('FP.CPI.TOTL.ZG','USA', range(1970, 2022), index= 'time', numericTimeKeys=True, labels=True).plot(figsize=(10, 6))
#plt.show()

data = wb.series.list(q='inflation')
data_list = []

choice = 0

for id in data:
    id_title = id['id']
    data_list.append(id_title)
    print(str(choice) + ": " + str(id['value']))
    choice = choice + 1

selection = int(input("Please enter the number of the database you'd like to chart."))

df = wb.data.DataFrame(data_list[selection], 'NGA', range(2015,2019), index='time', numericTimeKeys=True, labels=True)

fig = px.line(df, x='Time', y='NGA')
fig['layout']['xaxis']['autorange'] = 'reversed'
fig.show()

