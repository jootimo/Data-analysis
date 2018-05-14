import re

def parse(filename, type_cast=float):
    ''' Open file with name filename and parse comma separated values a into 2-d list
        
        @param    filename
        @param  type_cast   function that is used to cast the strings  into numbers
    '''
    data = []

    with open(filename, "r") as filestream:

        for line in iter(filestream.readline, ''):
            line = list(line.split(','))

            #Parse rows into lists
            for i in range(0, len(line)):

                #Remove newline characters
                line[i].replace('\n', '')

                line[i] = type_cast(line[i])

            data.append(line)

    return data


def parse_only_numbers(filename, type_cast):
    ''' 
    Open a file and parse it's numerical contents into a list
    
    @param  filename    name of the file to open
    @param  type_cast   function that is used to cast the strings  into numbers
    '''
    data = []

    with open(filename, "r") as filestream:

        for line in iter(filestream.readline, ''):
            line = list(line.split(','))
	
            #Parse rows into lists
            for i in range(0, len(line)):
            
				#Remove non numerical characters
                non_decimal = re.compile(r'[^\d.]+')
                line[i] = non_decimal.sub('', line[i])
				
                line[i] = type_cast(line[i])

            data.append(line)

    return data



