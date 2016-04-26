import re,ConfigParser

class BetterConfigParser(ConfigParser.SafeConfigParser):

    def get(self, section, option):
        result = ConfigParser.SafeConfigParser.get(self, section, option, raw=True)
        result = self.__replaceSectionwideTemplates(result)
        return result

    def items(self, section):
        list = ConfigParser.SafeConfigParser.items(self, section=section, raw=True)
        result = [(key, self.__replaceSectionwideTemplates(value)) for key,value
        in list]
        return result

    def optionxform(self, optionstr):
        '''
        enable case sensitive options in .ini files
        '''
        return optionstr

    def __replaceSectionwideTemplates(self, data):
        '''
        replace <section|option> with get(section,option) recursivly
        '''
        result = data
        findExpression = re.compile(r"((.*)\<!(.*)\|(.*)\!>(.*))*")
        groups = findExpression.search(data).groups()
        if not groups == (None, None, None, None, None): # expression not matched
            result = self.__replaceSectionwideTemplates(groups[1])
            result += self.get(groups[2], groups[3])
            result += self.__replaceSectionwideTemplates(groups[4])
        return result
