import owlready2
from owlready2 import *


# sync_reasoner()
# get classes, property, instances in ontology model
def get_ontInformation_list(onto_path):
    onto = get_ontology(onto_path).load()
    stdo_classes = list(cls.name for cls in onto.classes())  # get the names of entities in ontology
    instances_list = list(instance.name for instance in onto.individuals())
    object_properties_list = list(object_property.name for object_property in onto.object_properties())
    data_properties_list = list(data_property.name for data_property in onto.data_properties())
    return stdo_classes, instances_list, object_properties_list, data_properties_list


# judge the seg_word is whether belong to the ontology model
def is_class_in_ontology(seg_word, onto_path):
    stdo_classes, instances_list, object_properties_list, data_properties_list = get_ontInformation_list(onto_path)
    if seg_word in stdo_classes:
        return 'in_class'
    elif seg_word in instances_list:
        return 'in_instance'
    elif seg_word in object_properties_list:
        return 'in_object_property'
    elif seg_word in data_properties_list:
        return 'in_data_property'
    else:
        return 'None'


# get the class that has the same hierarchy with given entity
# if no class, return self
def get_sibling_classes(onto, cls_string):
    given_entity = onto[cls_string]
    # Get the parent classes of the given entity
    parent_classes = given_entity.is_a
    # For each parent, retrieve its subclasses
    siblings = set()
    for instance in onto.individuals():
        if onto[cls_string] in instance.is_a:
            siblings.add(instance)
    # Remove the original class from the list of siblings
    siblings.discard(given_entity)
    if len(siblings) >= 1:
        siblings.discard(given_entity)
    if len(siblings) < 1:
        for parent in parent_classes:
            siblings.update(parent.subclasses())
    if len(siblings) < 1:
        siblings.add(onto[cls_string])
    induced_entities_list = list(cls.name for cls in siblings)
    comment_list = get_comment_list(onto, cls_string)
    if len(comment_list) > 0:
        induced_entities_list = comment_list
    return induced_entities_list


# get the object property that has same domain and range with given object property
# if no object property, return self
def object_property_has_same_domain_range(onto, object_property_string):
    given_object_property = onto[object_property_string]
    # Get the domain and range of the given object property
    domain = given_object_property.domain
    range_ = given_object_property.range
    # Find other object properties with the same domain and range
    similar_props = []
    for other_prop in onto.object_properties():
        if other_prop != given_object_property \
                and other_prop.domain == domain and other_prop.range == range_:
            similar_props.append(other_prop.name)
    if len(similar_props) == 0:
        similar_props.append(object_property_string)
    comment_list = get_comment_list(onto, object_property_string)
    if len(comment_list) > 0:
        similar_props = comment_list
    return similar_props


# get the data property that has the same domain with given data property
def data_property_has_same_domain(onto, data_property_string, seg_words_list):
    given_data_property = onto[data_property_string]
    # Get the domain of the given data property
    domain = given_data_property.domain
    seg_words_list_until_data_property = seg_words_list[0: seg_words_list.index(data_property_string)]
    stdo_classes = list(cls.name for cls in onto.classes())
    instances_list = list(instance.name for instance in onto.individuals())

    # if the length of domain more than 1; data property belong to multi class
    if len(domain) > 1:
        temp_domian_list = []
        temp_domain_distance_list = []
        seg_words_list_until_data_property = seg_words_list[0: seg_words_list.index(data_property_string)]
        domain_name_list = list(domain1.name for domain1 in domain)
        for j in range(0, len(seg_words_list_until_data_property)):
            if seg_words_list_until_data_property[j] in domain_name_list:
                temp_domian_list.append(seg_words_list_until_data_property[j])
                temp_domain_distance_list.append(len(seg_words_list_until_data_property) - j)
            elif seg_words_list_until_data_property[j] in instances_list:
                class_type = onto[seg_words_list_until_data_property[j]].is_a[0]
                if class_type.name in domain_name_list:
                    temp_domian_list.append(class_type.name)
                    temp_domain_distance_list.append(len(seg_words_list_until_data_property) - j)
        if len(temp_domian_list) == 1:
            domain = onto[temp_domian_list[0]]
        elif len(temp_domian_list) == 0:
            domain = domain
        else:
            domain = onto[temp_domian_list[-1]]
    # Collect data properties with the same domain
    matching_properties = []
    for prop in onto.data_properties():
        if prop != given_data_property:
            # Check if domain matches
            if domain in prop.domain:
                matching_properties.append(prop.name)
    if len(matching_properties) == 0:
        matching_properties.append(data_property_string)
    comment_list = get_comment_list(onto, data_property_string)
    if len(comment_list) > 0:
        matching_properties = comment_list
    return matching_properties


# get the instances that have same types with given class
def instance_has_same_domain(onto, instance_string):
    types = set(onto[instance_string].is_a)
    # Collect instances with the same types
    matching_instances = []
    for inst in onto.individuals():
        if inst != onto[instance_string]:
            # Check if the types match
            if set(inst.is_a) == types:
                matching_instances.append(inst.name)
    if len(matching_instances) == 0:
        matching_instances.append(instance_string)
    comment_list = get_comment_list(onto, instance_string)
    if len(comment_list) > 0:
        matching_instances = comment_list
    return matching_instances


# get the words that belong to the same class
def get_words_same_hierarchy_with_word(onto_path, string_is_in_ontology, word, seg_words_list):
    entity_list = []
    onto = get_ontology(onto_path).load()
    if string_is_in_ontology == 'in_class':
        entity_list = get_sibling_classes(onto, word)
    elif string_is_in_ontology == 'in_object_property':
        entity_list = object_property_has_same_domain_range(onto, word)
    elif string_is_in_ontology == 'in_data_property':
        entity_list = data_property_has_same_domain(onto, word, seg_words_list)
    elif string_is_in_ontology == 'in_instance':
        entity_list = instance_has_same_domain(onto, word)
    return entity_list


# get the comment of object
def get_comment_list(onto, name_string):
    comment_name_list = []
    object_onto = onto[name_string]
    object_onto_comment = object_onto.comment
    if len(object_onto_comment) > 0:
        comment_name_list = object_onto.comment[0].split('\n')
    return comment_name_list


# using ontology create triple
def create_triple_list_using_onto(onto_path):
    onto = get_ontology(onto_path).load()
    triple_string_list = []
    # name information
    stdo_classes, instances_list, object_properties_list, data_properties_list = get_ontInformation_list(onto_path)
    stdo_classes_chinese = stdo_classes[1:]
    for object_class_name in stdo_classes_chinese:
        for subclass in onto[object_class_name].subclasses():
            class_information = subclass.name + '属于' + object_class_name
            triple_string_list.append(class_information)
        comment_name_list = get_comment_list(onto, object_class_name)
        if len(comment_name_list) > 0:
            for class_comment in comment_name_list:
                class_comment_information = class_comment + '等同于' + object_class_name
                triple_string_list.append(class_comment_information)
    for object_property in object_properties_list:
        object_onto = onto[object_property]
        object_domains = object_onto.domain
        object_ranges = object_onto.range
        for object_domain in object_domains:
            for object_range in object_ranges:
                object_property_information = object_domain.name + object_property + object_range.name
                triple_string_list.append(object_property_information)
        comment_name_list = get_comment_list(onto, object_property)
        if len(comment_name_list) > 0:
            for class_comment in comment_name_list:
                class_comment_information = class_comment + '等同于' + object_property
                triple_string_list.append(class_comment_information)
    for data_property in data_properties_list:
        object_onto = onto[data_property]
        object_domains = object_onto.domain
        for object_domain in object_domains:
            data_property_information = object_domain.name + '具有' + data_property
            triple_string_list.append(data_property_information)
        comment_name_list = get_comment_list(onto, data_property)
        if len(comment_name_list) > 0:
            for class_comment in comment_name_list:
                class_comment_information = class_comment + '等同于' + data_property
                triple_string_list.append(class_comment_information)
    for instance in instances_list:
        object_onto = onto[instance]
        object_types = object_onto.is_a
        for object_type in object_types:
            instance_information = instance + '是一种' + object_type.name
            triple_string_list.append(instance_information)
        comment_name_list = get_comment_list(onto, instance)
        if len(comment_name_list) > 0:
            for class_comment in comment_name_list:
                class_comment_information = class_comment + '等同于' + instance
                triple_string_list.append(class_comment_information)
    return triple_string_list


# add list into a text file (add the ontology list into the training text)
# add the ontology list into the dictionary
def add_list2Text_file(text_list, textFilePath):
    # open file
    with open(textFilePath, 'a', encoding='utf-8') as f:
        # write elements of list
        for items in text_list:
            f.write('%s\n' % items)
        print("File written successfully")

    # close the file
    f.close()



if __name__ == '__main__':
    # ontology_path = 'E:/all_project/chuanqin/shieldTunnelDesignChinese1.owl'
    ontoPath = 'E:/all_project/chuanqin/shieldTunnelDesignChinese1.owl'
    onto = get_ontology(ontoPath).load()
    # stdo_classes, instances_list, object_properties_list, data_properties_list = get_ontInformation_list(onto)
    # tmp = [val for val in stdo_classes if val in data_properties_list]
    # print(tmp)
    # data_property_test = onto['交叉隧道']
    # print(data_property_test.is_a)
    # aa = data_property_test.is_a
    # print(aa[0].name)
    # a = [1, 2, 3, 4]
    # b = a[0:a.index(2)]
    # print(b)
    # information_list = []
    # object_name = '反应位移法'
    # object_onto = onto[object_name]
    # object_types = object_onto.is_a
    # for object_type in object_types:
    #         data_property_information = object_name + '是一种' + object_type.name
    #         information_list.append(data_property_information)
    # #
    # print(information_list)
    list_information = create_triple_list_using_onto(ontoPath)
    print(list_information)
    filePath='E:/pythonProject/transByML/usingHuggingfaceTrans/trainWord2Vect/result.txt'
    add_list2Text_file(list_information, filePath)
