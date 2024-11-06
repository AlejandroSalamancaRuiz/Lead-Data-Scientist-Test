

sentiment_clf_prompt = """ 

Identify the sentiment of the Member in the conversation.

These are some examples of conversations with their respective sentiment:

1.
Member: Hi, I'm having some trouble registering and logging in to my online service account. My member ID is MEM456789. 
Technical Support: Sorry to hear that, MEM456789. Can you please tell me more about the issue you're experiencing? What error message are you seeing?
Member: It just says "Invalid username or password" every time I try to log in. I've tried resetting my password multiple times, but it doesn't seem to be working.
Technical Support: I apologize for the inconvenience. Can you please try resetting your password again, and this time, make sure to use a combination of uppercase and lowercase letters, as well as numbers?
Member: (sighs) Fine. I've reset it again. But I'm still getting the same error message.
Technical Support: I understand your frustration. Let me try to look into this further. Can you please confirm your email address associated with the account?
Member: It's johndoe@email.com.
Technical Support: Thank you. I'm going to go ahead and check on the status of your account. (pause) It looks like there was an issue with your account activation. I'm going to go ahead and activate it for you now.
Member: (irritated) Why wasn't it activated in the first place? I've been trying to get this working for hours.
Technical Support: I apologize for the mistake. It's possible that there was a technical glitch. But I've activated your account now, and you should be able to log in successfully.
Member: (sighs) Okay... let me try again.
Technical Support: Please go ahead and try logging in, and I'll wait to see if you're able to access your account.
Member: (pause) Okay... I'm in. Finally.
Technical Support: Great! I'm glad we were able to resolve the issue. Is there anything else I can assist you with today?
Member: No, that's all. Thanks for your help, I guess.
Technical Support: You're welcome, MEM456789. Have a great day.

Sentiment: negative

2. 
Member: Hi, I'm calling to get a case pre-authorized for a surgery I'm scheduled to have next week. My name is Emily Wilson, and my member ID is MEM123456.
PA Agent: Thank you for calling PA Customer Care, Emily. Can you please confirm your date of birth and the name of your primary care physician so I can verify your eligibility?
Member: My date of birth is March 12, 1985, and my primary care physician is Dr. Smith.
PA Agent: Thank you, Emily. I've located your account. Can you please provide me with more details about the surgery you're scheduled to have? What is the procedure code, and who is the surgeon performing it?
Member: The procedure code is XYZ123, and the surgeon is Dr. Johnson.
PA Agent: Okay, thank you. Can you please hold for just a moment while I check on the eligibility of this procedure?
Member: Sure.
PA Agent: Thank you for holding, Emily. I've checked on the eligibility, and it looks like you do have coverage for this procedure. However, I need to inform you that there is a 20% coinsurance associated with this procedure. Are you aware of this?
Member: Yes, I was aware of that.
PA Agent: Okay, great. I'm going to go ahead and pre-authorize the case for you. You should receive a confirmation letter in the mail within the next 3-5 business days. Is there anything else I can assist you with today?
Member: No, that's all. Thank you for your help.
PA Agent: You're welcome, Emily. Is there anything else you'd like to discuss or any other questions you have regarding your benefits?
Member: No, that's all. Thank you.
PA Agent: Alright, Emily. Your case has been pre-authorized, and you should be all set for your surgery next week. If you have any other questions or concerns, please don't hesitate to reach out to us. Have a great day.
Member: Thank you. You too.
PA Agent: You're welcome, Emily. Goodbye.
Member: Goodbye.

Sentiment: neutral

3.
Member: Hi, I'm calling to schedule an appointment with a specialist. My name is Emily Wilson and my member ID is MEM123456.
Customer Support: Hi Emily, thank you for calling! Can you please tell me a little bit more about what you're looking for in a specialist visit? What's the reason for the appointment?
Member: I've been experiencing some back pain and I'd like to see an orthopedic specialist. I was hoping to get an appointment as soon as possible.
Customer Support: I'd be happy to help you with that, Emily. Let me just check on some availability. Can you please hold for just a moment?
Member: Sure, that's fine.
Customer Support: Okay, I'm back. I've checked on our specialist availability and I have a few options for you. We have Dr. Smith available on Wednesday of this week, or Dr. Johnson available on Friday. Both of them are highly rated orthopedic specialists. Would you prefer one of those options or would you like me to look further?
Member: That sounds great, thank you! I think I'd prefer Dr. Smith on Wednesday. What time is the appointment?
Customer Support: Dr. Smith has an opening at 2 PM on Wednesday. Would you like me to go ahead and schedule that appointment for you?
Member: Yes, that would be great, thank you!
Customer Support: Wonderful! I've scheduled the appointment for you. You'll receive a confirmation email with all the details. Is there anything else I can assist you with today?
Member: No, that's all. Thank you so much for your help!
Customer Support: You're welcome, Emily! It was my pleasure to assist you. Have a great day and we'll see you on Wednesday!
Member: Thank you, you too!
Customer Support: Is there anything else I can help you with before we end the call?
Member: No, that's all. Thanks again!
Customer Support: Great, Emily! Have a great day!

Sentiment: positive

"""

outcome_clf_prompt =  """         

Determine the conversation outcome.

These are some examples of conversations with their respective outcomes:

1.
Member: Hi, I'm having some trouble registering and logging in to my online service account. My member ID is MEM456789. 
Technical Support: Sorry to hear that, MEM456789. Can you please tell me more about the issue you're experiencing? What error message are you seeing?
Member: It just says "Invalid username or password" every time I try to log in. I've tried resetting my password multiple times, but it doesn't seem to be working.
Technical Support: I apologize for the inconvenience. Can you please try resetting your password again, and this time, make sure to use a combination of uppercase and lowercase letters, as well as numbers?
Member: (sighs) Fine. I've reset it again. But I'm still getting the same error message.
Technical Support: I understand your frustration. Let me try to look into this further. Can you please confirm your email address associated with the account?
Member: It's johndoe@email.com.
Technical Support: Thank you. I'm going to go ahead and check on the status of your account. (pause) It looks like there was an issue with your account activation. I'm going to go ahead and activate it for you now.
Member: (irritated) Why wasn't it activated in the first place? I've been trying to get this working for hours.
Technical Support: I apologize for the mistake. It's possible that there was a technical glitch. But I've activated your account now, and you should be able to log in successfully.
Member: (sighs) Okay... let me try again.
Technical Support: Please go ahead and try logging in, and I'll wait to see if you're able to access your account.
Member: (pause) Okay... I'm in. Finally.
Technical Support: Great! I'm glad we were able to resolve the issue. Is there anything else I can assist you with today?
Member: No, that's all. Thanks for your help, I guess.
Technical Support: You're welcome, MEM456789. Have a great day.

Outcome: issue resolved

2. 
Member: Hi, I'm calling to schedule an appointment with a specialist. My name is Emily Wilson and my member ID is MEM123456.
Customer Support: Thank you for calling us, Emily. Can I just confirm your date of birth to verify your account?
Member: It's August 12, 1985.
Customer Support: Thank you, Emily. What type of specialist are you looking to see and what date were you hoping for the appointment?
Member: I need to see a cardiologist and I was thinking maybe next Wednesday or Thursday?
Customer Support: Okay, let me check the availability of our cardiologists. (pause) It looks like Dr. Smith is available next Wednesday at 2 PM or Dr. Johnson is available next Thursday at 9 AM. Both are highly rated specialists.
Member: That sounds great. Can you tell me a bit more about Dr. Smith's background?
Customer Support: Dr. Smith has been with our network for over 10 years and has excellent patient reviews. She specializes in preventive cardiology and is known for her thorough approach.
Member: That sounds good. And what about Dr. Johnson?
Customer Support: Dr. Johnson is also highly experienced with a strong background in interventional cardiology. He's known for his ability to explain complex conditions in an easy-to-understand way.
Member: Hmm, I think I'm leaning towards Dr. Smith. But before I schedule, can you tell me if there are any additional fees for seeing a specialist?
Customer Support: Let me check on that for you. (pause) It looks like your plan covers specialist visits, but there may be a copay depending on your specific coverage. I can provide you with more detailed information if you'd like.
Member: That would be great, thank you.
Customer Support: I'd be happy to look into that further for you. However, I want to make sure I understand your coverage correctly. Can you please hold for just a moment while I check on a few more details?
Member: Okay, sure.
Customer Support: Emily, I apologize for the delay. It looks like we need a bit more information from you to accurately determine the copay amount. Can you please call us back tomorrow so we can look into this further?
Member: Okay, that sounds good. I'll call back tomorrow then.
Customer Support: Sounds good, Emily. We'll look forward to speaking with you tomorrow. Is there anything else I can assist you with today?
Member: No, that's all. Thank you for your help.
Customer Support: You're welcome, Emily. Have a great day and we'll talk to you tomorrow.
Member: You too, bye.
Customer Support: Bye.

Outcome: follow-up action needed

"""