import pokebase as pb
import pandas as pd
import copy, os
import streamlit as st

class Pokemon:
    def __init__(self):
        self.pokemon = None
        self.level = None
        self.name = None
        self.types = None
        self.moves = None
        self.effective_moves = None

    def __str__(self):
        return f"{self.name.capitalize()}, {self.display_type_info()}, lvl {self.level}"

    def validate_pokemon(self, name):
        pokemon = pb.pokemon(name)
        try:
            pokemon.species
            return pokemon
        except AttributeError:
            return None

    def create(self, name, level):
        pokemon = self.validate_pokemon(name)
        if pokemon is not None:
            self.pokemon = pokemon
            self.level = level
            self.name = pokemon.name
            self.types = pokemon.types
            self.moves = None
            self.effective_moves = None

    def update_level(self, level):
        self.level = level

    ### POKEMON MOVES ###
    def _is_detail_right_gen(self, detail, gen='firered-leafgreen'):
        return detail.version_group.name == gen
    def _get_move_details(self, move, gen='firered-leafgreen'):
        try:
            return next(det for det in move.version_group_details if self._is_detail_right_gen(det, gen=gen))
        except StopIteration:
            return None
    def _is_leveled_move(self, detail):
        if detail is None:
            return False
        else:
            return detail.move_learn_method.name == 'level-up'
    def _is_already_learned_move(self, detail, level):
        if detail is None:
            return False
        else:
            return detail.level_learned_at <= level
    def _filter_moves(self):
        moves = [m for m in self.pokemon.moves if self._is_leveled_move(self._get_move_details(m))]
        moves = [m for m in moves if self._is_already_learned_move(self._get_move_details(m), self.level)]
        
        return moves
    def _get_move_type(self, move):
        return move.move.type.name
    def _get_move_base_power(self, move):
        return move.move.power
    
    def get_move_info(self):
        moves = self._filter_moves()
        df = pd.DataFrame(data={
            'pokemon': self.name,
            'moves': [m.move.name for m in moves],
            'types': [self._get_move_type(m) for m in moves],
            'power': [self._get_move_base_power(m) for m in moves]
        })

        self.moves = df
        return df

    def _get_type_effective_moves(self, pokemon_type):
        tt = pb.type_(pokemon_type.type.name)
        effective_moves = {}
        effective_moves.update({move_type.name: 'super effective' for move_type in tt.damage_relations.double_damage_from})
        effective_moves.update({move_type.name: 'not very effective' for move_type in tt.damage_relations.half_damage_from})
        effective_moves.update({move_type.name: 'no effect' for move_type in tt.damage_relations.no_damage_from})
        return {pokemon_type.type.name: effective_moves}
    def get_all_type_effective_moves(self):
        effective_moves = {}
        for t in self.types:
            effective_moves.update(self._get_type_effective_moves(t))

        self.effective_moves = effective_moves
        return effective_moves



    ### DISPLAY STUFF ###
    def display_type_info(self):
        return '/'.join([t.type.name.capitalize() for t in self.pokemon.types])

    def display_type_effective_against(self):
        super_effective_against = ', '.join([', '.join([t.name for t in t.type.damage_relations.double_damage_to]) for t in self.types])
        return super_effective_against

    def display_type_weakness(self):
        weak_against = ', '.join([', '.join([t.name for t in t.type.damage_relations.double_damage_from]) for t in self.types])
        return weak_against


class Stable:
    def __init__(self, path):
        self.pokes = []
        self.belt = []
        self.belt_moves = pd.DataFrame(columns=['pokemon'])
        self.load(path)

    def print(self):
        for p in self.pokes:
            print(p)

    ### SAVE & RELOAD STATE ###
    def save(self, path='./stable.csv'):
        df = self.display()
        df.to_csv(path, index=False)

    def load(self, path):
        if os.path.exists(path):
            saved_pokes = pd.read_csv(path)
            for ix, row in saved_pokes.iterrows():
                self.add(row['name'].lower(), row['level'])


    ### INVENTORY MANAGEMENT ###
    def add(self, pokemon_name, level):
        pokemon = Pokemon()
        pokemon.create(pokemon_name, level)
        if pokemon.pokemon is not None:
            self.pokes.append(pokemon)
        else:
            return 'Check pokemon name'
    def remove(self, pokemon_name):
        self.pokes = [p for p in self.pokes if p.name != pokemon_name]
        self.remove_from_belt(pokemon_name)
    def remove_all(self):
        self.pokes = []
        self.belt = []
        self.belt_moves = pd.DataFrame(columns=['pokemon'])

    def select(self, name):
        return next(p for p in self.pokes if p.name.lower() == name.lower())
    def add_to_belt(self, pokemon_name, pokemon_lvl=None):
        # validate the name first
        pokemon = Pokemon().validate_pokemon(pokemon_name)
        if pokemon is not None:

            # add to the stable if not already added
            try:
                self.select(pokemon_name)
            except StopIteration:
                self.add(pokemon_name, pokemon_lvl)

            # add to the belt
            pokemon = self.select(pokemon_name)
            self.belt.append(pokemon)

            # fetch pokemon moves
            pokemon.get_move_info()
            pokemon.get_all_type_effective_moves()
            self.belt_moves = pd.concat([
                self.belt_moves,
                pokemon.moves
            ], ignore_index=True)

        else:
            return 'Check pokemon name'
    def remove_from_belt(self, pokemon_name):
        self.belt = [p for p in self.belt if p.name != pokemon_name]
        self.belt_moves = self.belt_moves.query('pokemon != @pokemon_name')
    def add_all_to_belt(self):
        for p in self.pokes:
            self.add_to_belt(p.name)
    def remove_all_from_belt(self):
        for p in self.pokes:
            self.remove_from_belt(p.name)

    def update(self, name, level):
        try:
            p = self.select(name)
            p.update_level(level)
        except StopIteration:
            st.error('Pokemon not yet in stable')



    ### GENERATE ATTACK MATCHUPS ### 
    def generate_attack_matchups(self, moves, opposing_poke):
        attack_df = moves.copy()

        new_cols = []
        for poke_type, mapping in opposing_poke.effective_moves.items():
            new_col = f'vs {poke_type}'
            new_cols.append(new_col)
            attack_df[new_col] = attack_df['types'].map(mapping).fillna('normal effectiveness')

        return attack_df, new_cols
    def filter_attack_matchups(self, df, compare_cols, effectiveness='super'):

        if effectiveness == 'super': 
            df = df.loc[
                (df['power'].notnull())
                & ((df[compare_cols] == 'super effective').any(axis=1))
            ]

        elif effectiveness == 'normal':
            df = df.loc[
                (df['power'].notnull())
                & ((df[compare_cols] != 'no effect').all(axis=1))
                & ((df[compare_cols] != 'not very effective').all(axis=1))
                & ((df[compare_cols] != 'super effective').all(axis=1))
            ]

        return df


    ### GENERATE DEFENSE MATCHUPS ###

    def generate_defense_matchup(self, opponent_poke):
        
        # compare opponent's moves to each pokemon's type
        all_compare_cols = []
        defense_df = pd.DataFrame()
        for p in self.belt:
            poke_defense, compare_cols = self.generate_attack_matchups(opponent_poke.moves, p)
            all_compare_cols += compare_cols
            poke_defense.insert(0, 'defender', p.name)
            defense_df = pd.concat([defense_df, poke_defense], ignore_index=True)

        # label effectiveness
        defense_df.drop('pokemon', axis=1, inplace=True)
        defense_df.rename(columns={'moves': 'opposing move'}, inplace=True)
        defense_df.insert(0, 'overall effect', 'defender strong')
        defense_df['overall effect'] = (
            defense_df['overall effect']
            .mask(
                (defense_df[all_compare_cols] == 'super effective').any(axis=1),
                'defender weak')
            .mask(
                (defense_df[all_compare_cols] == 'normal effectiveness').any(axis=1),
                'normal effectiveness')
            .mask(
                (defense_df[all_compare_cols] == 'not very effective').any(axis=1),
                'defender effective')
        )

        defense_df = defense_df.loc[defense_df['power'].notnull()]
        
        return defense_df

    def filter_defense_matchups(self, df, defender_effectiveness='super'):

        if defender_effectiveness == 'super':
            df = df.loc[df['overall effect'].isin(['defender strong', 'defender effective'])]

        elif defender_effectiveness == 'normal':
            df = df.loc[df['overall effect'] == 'normal effectiveness']

        elif defender_effectiveness == 'weak':
            df = df.loc[df['overall effect'] == 'defender effective']

        return df

    def _assess_opponent_effectiveness(self, l):
        if 'defender weak' in l:
            defense_to_opponent = 'weak'
        elif 'normal effectiveness' in l:
            defense_to_opponent = 'normal'
        elif 'defender effective' in l:
            defense_to_opponent = 'strong'
        elif 'defender strong' in l:
            defense_to_opponent = 'super'
        return defense_to_opponent

    def aggregate_defense_matchups(self, df):
        return df.groupby('defender')['overall effect'].agg(list).apply(self._assess_opponent_effectiveness).to_dict()



    ### DISPLAY ###
    def display(self, selection='stable'):
        if selection == 'stable':
            l = self.pokes
        elif selection == 'belt':
            l = self.belt

        # create datafrme
        if len(self.pokes) == 0:
            df = pd.DataFrame(columns=['name', 'type', 'level'])

        else:
            df = pd.DataFrame(data={
                'name': [p.name.capitalize() for p in l],
                'type': [p.display_type_info() for p in l],
                'level': [p.level for p in l]
            })

        return df


class RecentOpponents:
    def __init__(self, size):
        self.opponents = []
        self.max_size = size

    def is_empty(self):
        return len(self.opponents) == 0

    def is_full(self):
        return len(self.opponents) == self.max_size

    def add(self, pokemon):
        if self.is_full():
            self.opponents.pop(0)
        self.opponents.append(copy.copy(pokemon))



### INITIALIZE OBJECTS 

@st.cache(allow_output_mutation=True)
def create_stable(path):
    return Stable(path)

@st.cache(allow_output_mutation=True)
def initialize_opponent():
    return Pokemon()

@st.cache(allow_output_mutation=True)
def initialize_recent_opponents(size=3):
    return RecentOpponents(size)


### START APP

st.set_page_config(layout="wide")
st.title('Pokemon Evaluation')
stable_path = './stable.csv'
stable = create_stable(stable_path)
opponent_poke = initialize_opponent()
recent_opponents = initialize_recent_opponents()


### GET INPUTS

row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)

with row1_col1:

    st.subheader('Input a pokemon')
    with st.form('Add pokemon'):
        poke_name = st.text_input('Pokemon name')
        poke_lvl = st.number_input('Level', min_value=1)
        submitted_add = st.form_submit_button('Add to stable')
        submitted_remove = st.form_submit_button('Remove from stable')
        submitted_update = st.form_submit_button('Update level')
        submitted_add_belt = st.form_submit_button('Add to belt')
        submitted_remove_belt = st.form_submit_button('Remove from belt')
    if submitted_add:
        error = stable.add(poke_name, poke_lvl)
        if isinstance(error, str):
            st.error(error, icon='🚨')
    if submitted_remove:
        stable.remove(poke_name)
    if submitted_update:
        stable.update(poke_name, poke_lvl)
    if submitted_add_belt:
        error = stable.add_to_belt(poke_name, poke_lvl)
        if isinstance(error, str):
            st.error(error, icon='🚨')
    if submitted_remove_belt:
        stable.remove_from_belt(poke_name)

with row1_col2:
    st.subheader('Current captured pokemon')
    
    # button to remove everyone from the stable
    if st.button('Remove all from stable'):
        # st.write(dir(Stable))
        stable.remove_all()

    # button to save current stable
    if st.button('Save current stable'):
        stable.save(stable_path)

    # display current stable
    stable_df = stable.display()
    st.dataframe(stable_df, use_container_width=True)

with row1_col3:
    st.subheader('Pokemon to compare')
    
    # add/remove all from the belt
    if st.button('Add all to belt'):

        add_progress = st.progress(0)
        for i, pokemon in enumerate(stable.pokes):
            stable.add_to_belt(pokemon.name)
            add_progress.progress((i+1)/len(stable.pokes))

        # stable.add_all_to_belt()
    if st.button('Remove all from belt'):
        stable.remove_all_from_belt()

    # display current belt
    belt_df = stable.display(selection='belt')
    st.dataframe(belt_df, use_container_width=True)

with row1_col4:
    st.subheader('Input opposing pokemon')

    with st.form('Opposing pokemon pokemon'):
        poke_name = st.text_input('Pokemon name')
        poke_lvl = st.number_input('Level', min_value=1)
        op_add = st.form_submit_button('Submit opponent')

    if op_add:
        opponent_poke.create(poke_name, poke_lvl)
        opponent_poke.get_move_info()
        opponent_poke.get_all_type_effective_moves()
        recent_opponents.add(opponent_poke)

    if not recent_opponents.is_empty():
        st.caption('Recent opponents:')
        for p in recent_opponents.opponents:
            if st.button(str(p)):
                opponent_poke = copy.copy(p)
                # st.write('update opponent')




### MAKE COMPARISONS
row2_col1, row2_col2 = st.columns(2)

# pre-compute matchups
if len(stable.belt) > 0 and opponent_poke.name is not None:
    attack_df, compare_cols = stable.generate_attack_matchups(stable.belt_moves, opponent_poke)
    defense_df = stable.generate_defense_matchup(opponent_poke)
    defense_mapping = stable.aggregate_defense_matchups(defense_df)
    attack_df['defense to opponent moves'] = attack_df['pokemon'].map(defense_mapping)

with row2_col1:

    st.header('Attack Strategy')
    if opponent_poke.name is not None:
        st.subheader(f'vs {str(opponent_poke)}')
        st.markdown(f'{opponent_poke.name.capitalize()} weak to: {opponent_poke.display_type_weakness()}')

    show_normal = st.checkbox('Show normal effectiveness moves', value=True)
    show_all = st.checkbox('Show all moves')


    if len(stable.belt) > 0 and opponent_poke.name is not None:

        st.caption('Super effective moves')
        st.dataframe(stable.filter_attack_matchups(attack_df, compare_cols, effectiveness='super'), use_container_width=True)

        if show_normal:
            st.caption('Normal effectiveness moves')
            st.dataframe(stable.filter_attack_matchups(attack_df, compare_cols, effectiveness='normal'), use_container_width=True)

        if show_all:
            st.caption('All moves')
            st.dataframe(stable.filter_attack_matchups(attack_df, compare_cols, effectiveness='all'), use_container_width=True)

with row2_col2:

    st.header('Defense Strategy')
    if opponent_poke.name is not None:
        st.subheader(f'vs {str(opponent_poke)}')
        st.markdown(f'{opponent_poke.name.capitalize()} effective against: {opponent_poke.display_type_effective_against()}')

    show_def_normal = st.checkbox('Show normal effectiveness attackers', value=True)
    show_def_weak = st.checkbox('Show attackers weak to the defender')

    if len(stable.belt) > 0 and opponent_poke.name is not None:
        st.caption('Super effectiveness against the opponent')
        st.dataframe(stable.filter_defense_matchups(defense_df, defender_effectiveness='super'), use_container_width=True)

        if show_def_normal:
            st.dataframe(stable.filter_defense_matchups(defense_df, defender_effectiveness='normal'), use_container_width=True)

        if show_def_weak:
            st.dataframe(stable.filter_defense_matchups(defense_df, defender_effectiveness='weak'), use_container_width=True)

        st.caption("Opponent's moves")
        st.dataframe(opponent_poke.moves)
