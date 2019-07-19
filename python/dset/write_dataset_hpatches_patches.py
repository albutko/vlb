import json

with open('./python/bench/hpatches/tasks/splits/splits.json','rb') as splits:
    splits = json.load(splits)
    for split in splits.keys():
        sequence_name_list = splits[split]["test"]
    # sequence_name_list = ['i_ajuntament', 'i_autannes', 'i_bologna', 'i_books', 'i_boutique', 'i_bridger',
    # 'i_brooklyn', 'i_castle', 'i_chestnuts', 'i_contruction', 'i_crownday', 'i_crownnight', 'i_dc',
    # 'i_dome', 'i_duda', 'i_fenis', 'i_fog', 'i_fruits', 'i_gonnenberg', 'i_greenhouse', 'i_greentea',
    # 'i_indiana', 'i_kions', 'i_ktirio', 'i_kurhaus', 'i_leuven', 'i_lionday', 'i_lionnight', 'i_londonbridge',
    # 'i_melon', 'i_miniature', 'i_nescafe', 'i_nijmegen', 'i_nuts', 'i_objects','i_parking','i_partyfood',
    # 'i_pencils', 'i_pinard', 'i_pool', 'i_porta', 'i_resort', 'i_salon', 'i_santuario', 'i_school', 'i_ski',
    # 'i_smurf', 'i_steps', 'i_table', 'i_tools', 'i_toy', 'i_troulos', 'i_veggies', 'i_village',
    # 'i_whitebuilding', 'i_yellowtent', 'i_zion', 'v_abstract', 'v_adam', 'v_apprentices', 'v_artisans',
    # 'v_astronautis', 'v_azzola', 'v_bark', 'v_bees', 'v_beyus', 'v_bip', 'v_bird', 'v_birdwoman',
    # 'v_blueprint', 'v_boat', 'v_bricks', 'v_busstop', 'v_calder', 'v_cartooncity', 'v_charing','v_churchill',
    # 'v_circus', 'v_coffeehouse', 'v_colors', 'v_courses','v_dirtywall', 'v_dogman', 'v_eastsouth',
    #  'v_feast', 'v_fest', 'v_gardens', 'v_grace', 'v_graffiti','v_home', 'v_laptop', 'v_london',
    #  'v_machines', 'v_man', 'v_maskedman', 'v_pomegranate', 'v_posters', 'v_samples', 'v_soldiers',
    #  'v_strand', 'v_sunseason', 'v_tabletop', 'v_talent', 'v_tempera', 'v_there', 'v_underground',
    #  'v_vitro','v_wall','v_wapping','v_war','v_weapons','v_woman','v_wormhole','v_wounded','v_yard','v_yuri']
        description_list = list()
        for s in sequence_name_list:
            if s[0] == 'i':
                description_list.append('Decreasing Light')
            else:
                description_list.append('Viewpoint angle')

        label_list = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5','t1','t2','t3','t4','t5']

        json_data = {}
        json_data['Dataset Name'] = 'HPatches (Split {})'.format(split)
        json_data['Description'] = 'Standard Benchmark'
        json_data['url'] = 'http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz'
        json_data['Sequence Number'] = len(sequence_name_list)
        json_data['Sequence Name List'] = sequence_name_list
        json_data['Sequences'] = []

        for idx, sequence_name in enumerate(sequence_name_list):
            sequence = {}
            sequence['Name'] = sequence_name
            sequence['Description'] = sequence_name
            sequence['Label'] = description_list[idx]
            sequence['Images'] = []
            sequence['Image Number'] = 16

            for image_idx, image_label in enumerate(label_list):
                image = {}
                image['file'] = '{}/{}.png'.format(sequence_name,image_label)
                image['id'] = str(image_label)
                image['label'] = str(image_label)
                sequence['Images'].append(image)

            sequence['Link Number'] = 15
            sequence['Links'] = []
            for label in label_list[1:]:
                link = {}
                link['source'] = 'ref'
                link['target'] = label
                link['file'] = '{}/H_ref_{}'.format(sequence_name, label[-1])
                sequence['Links'].append(link)
            json_data['Sequences'].append(sequence)

        with open('./datasets/dataset_info/{}_split_{}.json'.format('hpatches_patches', split),'w') as json_file:
            json.dump(json_data, json_file, indent=2)


sequence_name_list = ['i_ajuntament','v_yuri']
description_list = list()
for s in sequence_name_list:
    if s[0] == 'i':
        description_list.append('Decreasing Light')
    else:
        description_list.append('Viewpoint angle')

label_list = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5','t1','t2','t3','t4','t5']

json_data = {}
json_data['Dataset Name'] = 'HPatches (Split {})'.format('small')
json_data['Description'] = 'Standard Benchmark'
json_data['url'] = 'http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz'
json_data['Sequence Number'] = len(sequence_name_list)
json_data['Sequence Name List'] = sequence_name_list
json_data['Sequences'] = []

for idx, sequence_name in enumerate(sequence_name_list):
    sequence = {}
    sequence['Name'] = sequence_name
    sequence['Description'] = sequence_name
    sequence['Label'] = description_list[idx]
    sequence['Images'] = []
    sequence['Image Number'] = 16

    for image_idx, image_label in enumerate(label_list):
        image = {}
        image['file'] = '{}/{}.png'.format(sequence_name,image_label)
        image['id'] = str(image_label)
        image['label'] = str(image_label)
        sequence['Images'].append(image)

    sequence['Link Number'] = 15
    sequence['Links'] = []
    for label in label_list[1:]:
        link = {}
        link['source'] = 'ref'
        link['target'] = label
        link['file'] = '{}/H_ref_{}'.format(sequence_name, label[-1])
        sequence['Links'].append(link)
    json_data['Sequences'].append(sequence)

with open('./datasets/dataset_info/{}_split_small.json'.format('hpatches_patches'),'w') as json_file:
    json.dump(json_data, json_file, indent=2)
