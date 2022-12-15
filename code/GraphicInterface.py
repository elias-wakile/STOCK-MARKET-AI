import datetime
from tkinter import *
from tkcalendar import Calendar
from tkinter import ttk, PhotoImage, messagebox
import datetime
import yfinance as yf
import AI_NeuralNetwork_Trader


def create_entry_window():
    """
    create window for the gui
    :return: when all the values are valid return to stop the gui
    """
    stock_symbol = ["AAPL", "GOOGL", "MSFT", "INTC"]

    class Checkbar(Frame):
        """
        class that represent Checkbar
        """

        def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
            Frame.__init__(self, parent)
            self.vars = []
            for pick in picks:
                var = IntVar()
                chk = Checkbutton(self, text=pick, variable=var)
                chk.pack(side=side, anchor=anchor, expand=YES)
                self.vars.append(var)

        def state(self):
            """ Return the selected Checkbar"""
            return map((lambda var: var.get()), self.vars)

    def display_msg():
        """
        When press the submit butane check if all the selected values are valid and send to run the models
        :return: if on of the selected values is invalid return to get new one
        """
        format = '%m/%d/%y'
        sd = datetime.datetime.strptime(start_date.get_date(), format)
        ed = datetime.datetime.strptime(end_date.get_date(), format)
        delte_date = ed - sd
        if delte_date.days < 7:
            messagebox.showerror("Date error", "Please select two dates with at least a difference of at least 7 days.")
            return
        chose_stock = list(stock_entry.state())
        stock_names = []
        for i in range(len(chose_stock)):
            if chose_stock[i] == 1:
                stock_names.append(stock_symbol[i])

        if len(stock_names) == 0:
            messagebox.showerror("No Stock select error", "Please select at least two stock from the list")
            return
        window.destroy()
        AI_NeuralNetwork_Trader.main_def(sd, ed, stock_names)

    window = Tk()
    window.title('The Wall Street Orca')
    window.geometry('600x400+50+50')
    window.resizable(False, False)
    window.minsize(650, 650)
    icon = PhotoImage(file="icon.png")
    window.iconphoto(False, icon)
    logo = PhotoImage(file="Logo.png")
    Label(window, image=logo).grid(row=0, column=0, columnspan=2)
    start_date = Calendar(window, selectmode='day',
                          maxdate=datetime.date.today(),
                          mindate=datetime.date.today() - datetime.timedelta(days=59),
                          state='normal')
    start_label = Label(window, text='Begin Investment On:')
    start_label.grid(row=1, column=0, padx=20, pady=10)
    start_date.grid(row=2, column=0, padx=35, pady=10)
    end_date = Calendar(window, selectmode='day',
                        maxdate=datetime.date.today(),
                        mindate=datetime.date.today() - datetime.timedelta(days=59),
                        state='normal')
    end_label = Label(window, text='End Investment On:')
    end_label.grid(row=1, column=1, padx=20, pady=10)
    end_date.grid(row=2, column=1, padx=35, pady=10)
    stock_label = Label(window, text='Please select at last two stock from the list')
    stock_entry = Checkbar(window, ["Apple", "Google", "Microsoft", "Intel"])
    stock_label.grid(row=3, column=0, columnspan=2, pady=10, padx=10)
    stock_entry.grid(row=4, column=0, columnspan=2, pady=0)
    stock_entry.config(relief=GROOVE, bd=2)
    ttk.Button(window, text='Submit', command=display_msg).place(relx=0.825, rely=0.9)
    window.mainloop()


if __name__ == '__main__':
    create_entry_window()
