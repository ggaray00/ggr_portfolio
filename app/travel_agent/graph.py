from langgraph.checkpoint.memory import MemorySaver
from .builder import builder


# def get_langgraph(anthropic_api_key):
#     builder = get_builder(anthropic_api_key=anthropic_api_key)

memory = MemorySaver()
part_4_graph = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    interrupt_before=[
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",
    ],
)
    # return part_4_graph


# thread_id = str(uuid.uuid4())
# config = {
#     "configurable": {
#         # The passenger_id is used in our flight tools to
#         # fetch the user's flight information
#         "passenger_id": "3442 587242",
#         # Checkpoints are accessed by thread_id
#         "thread_id": thread_id,
#     }
# }