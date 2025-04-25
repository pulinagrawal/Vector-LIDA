##This was the original adaptive agent code, then I made more modular to make it easier to test.

#import time
#import traceback
#from pathlib import Path
#from types import SimpleNamespace
#
#import torch
#from agent import (FilmEnvironment, compute_average_embedding, vision_processor,
#                   throwing_ref_node, other_ref_node, lida_agent, Node)
#
#CONFIDENCE_THRESHOLD = 0.2
#
#def update_centroid(embeddings_list, new_embedding):
#    embeddings_list.append(new_embedding)
#    return compute_average_embedding(embeddings_list)
#
#def adaptive_agent(environment, lida_agent, steps=1000):
#    lida_agent = SimpleNamespace(**lida_agent)
#
#    # Initialize centroid pools from reference images
#    throwing_embeddings = environment.collect_embeddings("throwing")
#    not_throwing_embeddings = environment.collect_embeddings("not_throwing")
#
#    category_embeddings = {
#        "throwing": compute_average_embedding(throwing_embeddings),
#        "not_throwing": compute_average_embedding(not_throwing_embeddings)
#    }
#
#    throwing_ref_node.features = category_embeddings["throwing"]
#    other_ref_node.features = category_embeddings["not_throwing"]
#
#    for step in range(steps):
#        frame = environment.receive_sensory_stimuli()
#        if frame is None:
#            break
#
#        associated_nodes = lida_agent.sensory_system.process(frame)
#        current_features = vision_processor(frame)
#        if not current_features:
#            continue
#
#        current_node = current_features[0]
#        throwing_sim = throwing_ref_node.similarity_function(throwing_ref_node, current_node)
#        not_throwing_sim = other_ref_node.similarity_function(other_ref_node, current_node)
#
#        confidence = abs(throwing_sim - not_throwing_sim)
#        decision = "throwing" if throwing_sim > not_throwing_sim else "not_throwing"
#
#        print(f"Step {step}: Throwing={throwing_sim:.3f}, NotThrowing={not_throwing_sim:.3f}, Confidence={confidence:.3f}, Decision={decision}")
#
#        # Log and possibly update centroids
#        if confidence > CONFIDENCE_THRESHOLD:
#            if decision == "throwing":
#                category_embeddings["throwing"] = update_centroid(throwing_embeddings, current_node)
#                throwing_ref_node.features = category_embeddings["throwing"]
#                print("  → Updated 'throwing' centroid")
#            else:
#                category_embeddings["not_throwing"] = update_centroid(not_throwing_embeddings, current_node)
#                other_ref_node.features = category_embeddings["not_throwing"]
#                print("  → Updated 'not throwing' centroid")
#
#        # Create decision node for LIDA perception
#        decision_node = Node(content=decision.replace("_", " "), activation=0.99)
#        associated_nodes.append(decision_node)
#
#        # Run through LIDA architecture
#        lida_agent.csm.run(associated_nodes)
#        winning_coalition = lida_agent.gw.run(lida_agent.csm)
#
#        motor_commands = {"record": None}
#        if hasattr(winning_coalition, 'nodes'):
#            for node in winning_coalition.nodes:
#                if hasattr(node, 'content'):
#                    if node.content == "throwing":
#                        motor_commands = {"record": 0}
#                        break
#                    elif node.content == "not throwing":
#                        motor_commands = {"record": 1}
#                        break
#
#        selected_behavior = lida_agent.procedural_system.run(winning_coalition)
#        if selected_behavior:
#            behavior_commands = lida_agent.sensory_motor_system.run(selected_behavior, associated_nodes, winning_coalition)
#            behavior_commands = lida_agent.sensory_motor_system.get_motor_commands()
#            if behavior_commands and behavior_commands['record'] is not None:
#                motor_commands = behavior_commands
#
#        environment.run_commands(motor_commands)
#
#if __name__ == '__main__':
#    try:
#        env = FilmEnvironment(
#            video_source=0,
#            reference_image_folders=["throwing", "not_throwing"],
#            output_dir="film_agent/recordings",
#            fps=30,
#            display_frames=False
#        )
#
#        adaptive_agent(env, lida_agent, steps=1000)
#
#    except KeyboardInterrupt:
#        print("Interrupted by user, shutting down...")
#    except Exception as e:
#        print(f"ERROR: {e}")
#        traceback.print_exc()
#    finally:
#        print("Cleaning up resources...")
#        with env.thread_lock:
#            env.thread_active = False
#            env.should_record = False
#            env.is_recording = False
#            with env.recording_queue.mutex:
#                env.recording_queue.queue.clear()
#        env.close()
#        print("Done!")
