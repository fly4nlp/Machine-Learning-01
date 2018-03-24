from openpyxl import load_workbook

def read_excel(path):
    my_workbook = load_workbook(path)
    work_sheet = my_workbook.get_sheet_by_name('Sheet1')

    result = []
    header = []
    is_first_line = True
    for row in work_sheet.rows:
        if is_first_line:
            for cell in row:
                header.append(cell.value)
            is_first_line = False
        else:
            column_index = 0
            temp = {}
            for cell in row:
                temp[header[column_index]] = cell.value
                column_index += 1
            result.append(temp)
    return result

