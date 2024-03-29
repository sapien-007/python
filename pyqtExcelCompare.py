import sys
from PyQt5 import QtWidgets, uic
import datetime
import pytz
import boto3
import os
import subprocess
import json
import pandas as pd
from pathlib import Path

qtcreator_file  = "xl_compare.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.compare_xl.clicked.connect(self.CompareXL)

    def CompareXL(self):
            f1 = Path(self.xl1_box.toPlainText())
            f2= Path(self.xl2_box.toPlainText())

            # get index col from data
            df = pd.read_excel(f2)
            index_col = df.columns[0]


            df_OLD = pd.read_excel(f1, index_col=index_col).fillna(0)
            df_NEW = pd.read_excel(f2, index_col=index_col).fillna(0)

            # Perform Diff
            dfDiff = df_NEW.copy()
            droppedRows = []
            newRows = []

            cols_OLD = df_OLD.columns
            cols_NEW = df_NEW.columns
            sharedCols = list(set(cols_OLD).intersection(cols_NEW))

            for row in dfDiff.index:
                if (row in df_OLD.index) and (row in df_NEW.index):
                    for col in sharedCols:
                        value_OLD = df_OLD.loc[row,col]
                        value_NEW = df_NEW.loc[row,col]
                        if value_OLD==value_NEW:
                            dfDiff.loc[row,col] = df_NEW.loc[row,col]
                        else:
                            dfDiff.loc[row,col] = ('{}-->{}').format(value_OLD,value_NEW)
                else:
                    newRows.append(row)

            for row in df_OLD.index:
                if row not in df_NEW.index:
                    droppedRows.append(row)
                    dfDiff = dfDiff.append(df_OLD.loc[row,:])

            dfDiff = dfDiff.sort_index().fillna('')
            print(dfDiff)
            print('\nNew Rows:     {}'.format(newRows))
            print('Dropped Rows: {}'.format(droppedRows))

            # Save output and format
            fname = '{} vs {}.xlsx'.format(f1.stem,f2.stem)
            writer = pd.ExcelWriter(fname, engine='xlsxwriter')

            dfDiff.to_excel(writer, sheet_name='DIFF', index=True)
            df_NEW.to_excel(writer, sheet_name=f2.stem, index=True)
            df_OLD.to_excel(writer, sheet_name=f1.stem, index=True)

            # get xlsxwriter objects
            workbook  = writer.book
            worksheet = writer.sheets['DIFF']
            worksheet.hide_gridlines(2)
            worksheet.set_default_row(15)

            # define formats
            date_fmt = workbook.add_format({'align': 'center', 'num_format': 'yyyy-mm-dd'})
            center_fmt = workbook.add_format({'align': 'center'})
            number_fmt = workbook.add_format({'align': 'center', 'num_format': '#,##0.00'})
            cur_fmt = workbook.add_format({'align': 'center', 'num_format': '$#,##0.00'})
            perc_fmt = workbook.add_format({'align': 'center', 'num_format': '0%'})
            grey_fmt = workbook.add_format({'font_color': '#E0E0E0'})
            highlight_fmt = workbook.add_format({'font_color': '#FF0000', 'bg_color':'#B1B3B3'})
            new_fmt = workbook.add_format({'font_color': '#32CD32','bold':True})

            # set format over range
            ## highlight changed cells
            worksheet.conditional_format('A1:ZZ1000', {'type': 'text',
                                                    'criteria': 'containing',
                                                    'value':'-->',
                                                    'format': highlight_fmt})

            # highlight new/changed rows
            for row in range(dfDiff.shape[0]):
                if row+1 in newRows:
                    worksheet.set_row(row+1, 15, new_fmt)
                if row+1 in droppedRows:
                    worksheet.set_row(row+1, 15, grey_fmt)

            # save
            writer.save()
            self.results_window.setText(fname)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
