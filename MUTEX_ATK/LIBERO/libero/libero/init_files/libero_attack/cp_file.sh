tasks=("KITCHEN_SCENE1_put_the_black_bowl_on_the_plate" "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet" "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate" "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate" "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove" "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove" "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket" "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket" "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket" "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket" "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket" "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray" "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray" "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate" "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate" "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate" "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate")

for task in "${tasks[@]}"
do
	cp "../libero_90/${task}.pruned_init" .
done