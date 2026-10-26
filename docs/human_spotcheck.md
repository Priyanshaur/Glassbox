# Human spot-check (blind; rate by hand)

The rule-based numbers are circular: DPO was trained on the same keyword-coverage criterion it is scored on. This blind check asks whether a person prefers the DPO stories.

## How to rate

1. Read each item's prompt, then story **A** and story **B**. Which model wrote which is hidden and the order is shuffled.
2. Judge: (a) does it use all of the prompt's words naturally, (b) does it read like a coherent story, (c) is it free of repetition and gibberish. Write `A`, `B` or `tie`.
3. Stories are shown exactly as generated, with a 256-token limit, so some end abruptly. Rate what you read; do not guess which one is which.
4. Do not open the key file until every item is rated. It is deliberately not in git: `checkpoints/human_spotcheck_key.json` (the `checkpoints/` folder is git-ignored).
5. Afterwards unblind with the key (`A`/`B` map to `sft` or `dpo_s0`) and tally: how often the DPO story was preferred among non-ties, and ties.

Provenance: 20 of the 100 validation prompts, chosen at random (rng seed 20261027); one random sample per prompt, the same sample index for both models; SFT vs DPO lr 5e-6 seed 0 (kernel `glassbox-dpo` v3). Ratings below are left blank on purpose.

## Item 1

**Prompt:** Write a short story using all of these words: mean, walk, woman.

**Story A**

> Once upon a time there was a brave girl who was very brave. Every day she would go walking in the woods. As she walked, she would look down and see her feet sticking out like a stick.
>
> One day she decided to take a walk, so she walked and walked until she saw a big, beautiful tree. It was so big and she wanted to walk up it. As she stepped closer, she saw a man walking towards the tree.
>
> The man was very nice and asked the girl to come closer. He said he was very brave and would try to climb up the trunk. The girl was a bit scared but decided to try.
>
> When she got to the top of the tree she was surprised to find that the man was actually very nice. He had seen her brave heart and wanted to help her. He held out his hand and said, "I will help you if you want!"
>
> The girl was so happy and thanked the man. She carefully stepped down and waved goodbye before walking back home. From that day onwards, the girl was no longer scared of the manâ€™s mean ways. She had learned that being brave can help you to be brave and protect yourself.

**Story B**

> Once upon a time there was a brave girl named Lucy. She was always looking for ways to behave. One day she decided to try and cross a busy road. She knew it would be a terrible journey, but she wanted to make it.
>
> So, she turned around and started walking. She saw a beautiful castle and a big river, but she didn't want to cross it. She wanted to get to the other side.
>
> Then, she saw a big log. It was lying in the middle of a terrible river. Lucy had an idea. She started to throw the log into the river.
>
> Sure enough, the log sailed through the river and soon made it over the bridge. Lucy was so happy she crossed the bridge! She was proud of herself and she knew she could do anything.

**Rating (item 1):** preferred = ____ (A / B / tie)  |  notes: 

## Item 2

**Prompt:** Write a short story using all of these words: prize, right, started.

**Story A**

> Once upon a time, there was a little girl called Molly. Molly had a friend called Sam who was very generous. Every day, they would play together.
>
> One day, Sarah wanted to play a game. She said to Jack, â€œI have a prize for you. It is a teddy bear. You can take it home with you.â€
>
> Jack was so happy to take the prize home. He thanked Sarah for being so generous.
>
> When they got home, Jack and Sarah played the game together. They both won, but each had a turn. Jack won a few times, but he still liked the teddy bear.
>
> The next day, Molly came back to the game with her prize. She said to Sam, â€œI won a prize for you!â€
>
> But there was something else in the game. It was a game of 'A'. Bob had a lot of fun playing the game, but it was not as fun as Molly had hoped for.
>
> Sally reminded Jack that she had kept her prize for a very long time and it was all because she had tried so hard. Jack realised that he had a lot of fun with it and that it was not fair. He gave the prize back

**Story B**

> Once upon a time, there was a little girl called Molly. She had a prize that she wanted to complete. It was a big, shiny prize. But, Molly didn't know what it was.
>
> So, Molly asked her mom. Her mom said, "It's a prize, Molly. You can finish the prize." Molly was so excited!
>
> Molly worked very hard with her prize. When she was done, she was so happy. Her prize looked beautiful.
>
> Molly was so proud of her prize that she hugged it. She knew she had worked very hard to complete it. Now Molly was sure that she would win the prize!

**Rating (item 2):** preferred = ____ (A / B / tie)  |  notes: 

## Item 3

**Prompt:** Write a short story using all of these words: octopus, sound, tentacles.

**Story A**

> Once upon a time there was a curious little girl named Sarah. She was always asking questions and wanted to learn everything she could. One day, while exploring, she came across an octopus. Sarah gasped in awe at the incredible octopus.
>
> "Hello there," Sarah said. "What are you doing?"
>
> The octopus frowned. "I'm stuck around in the ocean," it said.
>
> Sarah was intrigued. "What's a problem?"
>
> "The ocean is too deep and I don't want to get involved," the octopus said. "I'm scared to go near the ocean."
>
> Sarah thought for a moment. "I can help you," she said. "I can make sure you stay safe and stay away from the deep ocean. Then, I'll help you get back to the shore."
>
> The octopus smiled. "Thank you," it said. "I'll be safe."
>
> With Sarah's help, the octopus was soon safely back on the shore. From that day forward, Sarah was no longer scared. She learned a valuable lesson about being careful in the ocean.

**Story B**

> Ben and Lily were playing in the sand near the sea. They liked to make shapes and towers with their hands and shells. They also liked to splash each other with water and make funny noises.
>
> "Look, I can make a big octopus with eight arms!" Ben said, using a stick and a towel.
>
> "Wow, that's cool, Ben! Can I try?" Lily asked, reaching for the stick.
>
> "Sure, here you go!" Ben said, handing her the stick.
>
> Lily used the stick to make a tail and two small arms. Then she stuck her head out of the stick and looked at Ben.
>
> "Hi, I'm Lily. Do you want to play with me?" she said, giggling.
>
> "Hi, I'm Ben. Yes, I want to play with you. What do you want to play?" Ben said, wiping his eyes.
>
> "I want to play a game of cards with the water weap. We can splash each other and try to catch the fish with the octopus mouth." Lily said, smiling.
>
> "Okay, that sounds fun. Let's do it!" Ben said, nodding.
>
> They took turns throwing rocks into the water and

**Rating (item 3):** preferred = ____ (A / B / tie)  |  notes: 

## Item 4

**Prompt:** Write a short story using all of these words: didn, saying, true.

**Story A**

> Lily and Ben are friends. They like to play with toys and joke with each other. One day, they find a big box in the park. They open it and see many shiny things inside. There are rings, necklaces, bracelets, and earrings.
>
> "Wow, look at these!" Lily says. "They are so pretty. Can I have one?"
>
> "Sure, I have one," Ben says. "But only one ring. It is not ours. Maybe someone lost it."
>
> Lily looks at the rings. She does not understand what they mean. She thinks they are shiny and pretty. She wants to try them on.
>
> "Can I keep one?" she asks.
>
> "No, you can't," Ben says. "They are not ours. They are for everyone. You can find your own ring. It is for your birthday."
>
> Lily is sad. She really likes the ring. She thinks Ben is not being nice. She decides to take the ring from Ben. She holds it in her hand.
>
> "Hey, give it to me!" Ben says. He tries to grab the ring from Lily.
>
> "No, it's mine!" Lily says. She holds the ring

**Story B**

> Once upon a time there lived a little girl named Jane. Jane loved to explore, and she was always looking for something new. One day, she ventured down a path into the forest. As Jane ventured further and further into the woods, she saw a unique rock. She wanted to take it home with her.
>
> "Can I bring this home with me?" Jane asked her father.
>
> "No, that's not ours," her father said. "It's not ours.
>
> Jane was confused. She wanted that rock, so she tried to persuade her father to let her.
>
> "Please daddy," she said, but her father refused. He knew it was valuable, so he said no again.
>
> Jane began to cry, but she had no choice. She just walked away with her head down, wishing for the rock to be hers.

**Rating (item 4):** preferred = ____ (A / B / tie)  |  notes: 

## Item 5

**Prompt:** Write a short story using all of these words: friend, shared, wanted.

**Story A**

> Sam and his little sister, Sophie, were so excited to have a new friend. They were having a great time playing together.
>
> But soon they started to quarrel. Sam wanted to play with Sophie's toys, but Sophie wanted to play with her dolls. They kept arguing and wouldn't listen to each other.
>
> Finally, Jake remembered that they were friends and he wanted to share with Sophie. So he gave Sophie some of his toys to play with. Sophie was so happy and they both shared a big hug.
>
> The two of them played together for a long time and had so much fun. Jake and Sophie learned that it is more important to be kind to each other, than to quarrel.

**Story B**

> Sam was a little boy who liked to play with his toys. He had a lot of cars, trains, and trains that he zoomed around his room. He also had a friend, Lily, who was a girl with curly hair and a big smile. They liked to play together and share their toys.
>
> One day, Sam and Lily found a big box in the garage. It had a picture of a plane on it and a picture of a plane on the side. Sam and Lily were curious and excited. They opened the box and saw many wires and tubes and buttons. They did not know what they did, but they thought they looked fun.
>
> They started to play with the wires and tubes and buttons and buttons. They made noises and moved them around. They did not know that their toy had a battery inside. When they pressed the red button, the battery made a loud noise and started to flash. Sam and Lily were scared and tried to stop it, but it was too late. The battery exploded and burned their hands and feet. They cried out in fear and tried to run away, but it was too late.
>
> Mom and Dad heard their screams and ran to the garage. They saw the fire and the smoke and the broken pieces.

**Rating (item 5):** preferred = ____ (A / B / tie)  |  notes: 

## Item 6

**Prompt:** Write a short story using all of these words: fake, play, wear.

**Story A**

> One day, Timmy and his friend Sally were playing in the park. Sally had a red ball that she loved to play with. They were having so much fun, but then Sally accidentally kicked the ball too hard. It went over the fence into a neighbor's yard.
>
> Sally started to cry because she thought she would never get her ball back. But Timmy said, "Don't worry, Sally. We can ask the neighbor if we can get it back."
>
> Timmy and Sally went to the neighbor's house and knocked on the door. The neighbor was nice and let them in. Timmy said, "Sally, did you kick your ball over the fence?"
>
> Sally said, "Yes, Timmy. I'm sorry. I didn't mean to kick it so hard."
>
> Timmy said, "It's okay, Sally. Let's go ask the neighbor if we can get your ball back." So they went over to their neighbor's house and asked if they could have the ball back. The neighbor said yes and Timmy and Sally were very happy. They thanked the neighbor and went back home to play.

**Story B**

> One day, Timmy and his friend Sally were playing in the park. Sally had a toy phone that she wanted to play with. She asked Timmy to help her make a fake sign. Timmy said he could help.
>
> In the park, they saw a big dog. The dog looked nice and Timmy said, "Let's play with the fake sign!" They put the fake sign in the sandbox and started to play with the fake sign. They made a pretend fire and threw sticks to the fake sign.
>
> After they played, they put the fake sign on the fake sign. Timmy and Sally pretended to be police and caught bad guys. They laughed and had fun playing in the park. It was a good day for Timmy and Sally.

**Rating (item 6):** preferred = ____ (A / B / tie)  |  notes: 

## Item 7

**Prompt:** Write a short story using all of these words: magazines, outside, time.

**Story A**

> Once upon a time, there was a little girl named Lily. She loved to skip around the park with her mommy. One day, Lily saw a big, fierce dog. She got scared and started to skip away, but her mommy said, "Don't be scared, Lily. The dog is nice."
>
> Later that day, Lily and her mommy went to the store. Lily saw a magazine with pictures of animals. She said, "Mommy, can we buy this magazine?" Her mommy said, "Yes, we can buy it. But first, let's go outside and skip around the park."
>
> Lily and her mommy skipped around the park, and Lily was happy. She said, "I like skipping, even if the dog is fierce. Can we buy the magazine when we get home?" Her mommy smiled and said, "Yes, we can buy it when we get home."

**Story B**

> Once upon a time, there was a little girl named Lili. She was three years old and loved to play outside in her garden. One day, she was in the garden looking up at the magazine. It was very big and she liked looking at the pictures. She was feeling very calm and happy.
>
> Suddenly, the magazine began to move! Lili was surprised and quickly looked inside. She was amazed to see a huge magazine!
>
> Lili wanted to keep the magazine, but she knew she shouldn't. She was feeling a bit ashamed, so she decided to leave it in the garden.
>
> The next day, Lili was playing in the garden again. Suddenly, the magazine started to move! Across the garden of the big magazine, Lili saw a little fairy! The fairy was very friendly and asked Lili to come and show her the magazine.
>
> Lili was so excited! She took the magazine and showed it to the fairy. The fairy was very happy to see the magazine and thanked Lili for showing it to her.
>
> Lili and the fairy became friends and they had lots of fun playing together. Lili was so happy that she had found

**Rating (item 7):** preferred = ____ (A / B / tie)  |  notes: 

## Item 8

**Prompt:** Write a short story using all of these words: outside, play, thinking.

**Story A**

> Sam and Lily were friends. They liked to play in the park. One day, they saw a man with a big sack. He was walking around the park and did not see them.
>
> "What is in the sack?" Sam asked Lily.
>
> "I don't know. Maybe toys or candy, or something good," Lily said.
>
> They went closer to the man and looked inside the sack. They saw many things, but they did not know what to find.
>
> "Maybe we should ask someone," Sam said.
>
> "Or maybe we should try to play with the sack," Lily said.
>
> They walked around the park and saw a lady with a dog. The dog was big and brown, and had a long tail.
>
> "Excuse me, lady," Sam said. "Can we go and see the sack?"
>
> The lady smiled and said, "Sure, but be careful. The man might not notice you."
>
> Sam and Lily went to the sack and opened it. Inside, they saw some cookies, some chips, some juice, and some cookies.
>
> "Wow, look at all those cookies!" Lily said.
>
> "They look yummy!" Sam said.
>
> They took

**Story B**

> Once upon a time there was a little girl. Her name was Sue. She was three years old.
>
> One day she decided to play in the garden. She was looking for something to do. Then she noticed a big open space.
>
> Sue said, "Wow! How about a game?"
>
> Her mom smiled and said, "How about a game where you have to find a ball?â€
>
> Sue was excited. She said, â€œYay,â€ and started thinking.
>
> She searched around the garden, looking for a ball. Then, she noticed a big, open field. She ran towards it and saw a big, bouncy ball.
>
> Sue was so excited. She said, "Yay! I found a ball!"
>
> She ran to the ball and started rolling it around the field. She was having a lot of fun.
>
> Then, her mom called from the house. "Sue, it's time for lunch!"
>
> Sue said goodbye to the ball and went inside. She was happy that she found such a big and open space to play with her new game.

**Rating (item 8):** preferred = ____ (A / B / tie)  |  notes: 

## Item 9

**Prompt:** Write a short story using all of these words: happy, stand, until.

**Story A**

> Once upon a time, there was a big fountain in the middle of a big park. A little duck was swimming in the fountain and it was very happy. All of a sudden, a big frog jumped in! The frog liked the fountain because it was big and green. It made a big splash and all the children wanted to play in the fountain.
>
> The little duck was happy and started to jump around. It was so excited to be in the fountain. All the children laughed and smiled as they watched the duck jump and splash. But then, all the sudden, the duck stopped jumping. It was a very sad day for the children.
>
> The duck just stayed in the fountain and looked around. Suddenly, it felt very sad. It looked around and it saw a big frog hopping near the fountain. The frog was very friendly and it hopped right up to the duck. The duck was so happy that it was able to make the frog happy.
>
> The duck and the frog became friends. The duck would come visit the pond every day and the frog would splash around in the fountain. They would play together and the duck would laugh and show the frog how much fun it was having.
>
> The duck and the frog lived happily together in the fountain

**Story B**

> Once upon a time, there was a big castle. Inside the castle, there was a king. He had a big horse and a nice man. One day, the king said to his horse, "I want to make the castle special." The horse said, "I want to help!" So the king and the horse went to the king's castle.
>
> The king used a big hammer to make the castle. The king used a hammer to fix the castle. The king said, "Thank you, horse. Now we can have a nice place to live." The horse said, "You're welcome, king. We make the castle even more pretty."
>
> The king and the horse became good friends. They played and explored the castle together. The king was very happy that he had the horse with him. They lived happily ever after.

**Rating (item 9):** preferred = ____ (A / B / tie)  |  notes: 

## Item 10

**Prompt:** Write a short story using all of these words: anymore, happy, trees.

**Story A**

> Mommy had a big bag of apples. She wanted to give one to her little girl, who was three years old. She was very excited to give them away.
>
> The little girl was very ignorant. She didn't know how to look at the apples. Mommy tried to explain to her what was happening but the little girl just kept saying no.
>
> Mommy got very angry and told her to put the apples back. But the little girl just kept saying no. Mommy was getting angry too.
>
> Finally, after a lot of talk and a lot of talking, the little girl finally agreed to put the apples back on the ground. Mommy was happy again and gave the little girl a big hug.
>
> The little girl was very happy that she could help Mommy with the apples. She was proud of herself for understanding the importance of looking after the apples.
>
> Mommy and the little girl hugged and said goodbye, and Mommy carried the bag of apples away.

**Story B**

> Mommy had a big bag of candy. She wanted to share with her little boy. He saw the bag and opened it. He was so excited to see the candy. He started to eat it.
>
> His Mommy said, "No, sweetheart, you can't eat too much candy. It's not good for your tummy."
>
> The little boy was sad. He wanted to eat more candy. He asked his Mommy, "Please, Mommy. Just a little bit. I won't eat too much candy."
>
> Mommy said, "No sweetheart, you already ate too much candy. You need to eat your own food."
>
> But the little boy kept asking for more candy. He didn't want to eat any. Then, Mommy said, "Let's go get something else. We can get some ice cream."
>
> The little boy cheered. He was happy to hear that. He said, "Yay! Ice cream!"
>
> The little boy and Mommy went to get ice cream. The little boy was very happy. Now he could eat candy and he didn't have to eat too much anymore. He was happy and healthy.

**Rating (item 10):** preferred = ____ (A / B / tie)  |  notes: 

## Item 11

**Prompt:** Write a short story using all of these words: green, searched, started.

**Story A**

> Once upon a time there was a boy called Tom. Tom was very curious. One day, he went to the park with his dad. Tom saw a big green tree. He wanted to climb it. So his dad helped him.
>
> Tom looked up at the tree and he saw a lot of leaves. He wanted to climb it, so he began to climb. His dad told him to be careful and watch his step.
>
> Tom was having fun climbing and the leaves were so green. He wanted to reach the top of the tree. Suddenly, his dad reminded him that he had left the leaves on the other side of the park.
>
> Tom quickly ran to the edge of the tree and he climbed up. He was so happy he got to the top. He wanted to stay and explore the park some more. So he decided to climb down the tree again.
>
> Tom had a great day exploring the park and he remembered to be careful while he climbed the tree.

**Story B**

> Once upon a time there was a boy called Tom. Tom was looking for something in the park. He looked under the bushes and behind the trees, but he couldn't find anything.
>
> He wondered what he could search for. Suddenly, he looked up at the sky and saw a green apple. He was so happy! He ran over to the apple, grabbed it and took a big bite.
>
> It was so juicy and delicious. Tom was very happy and decided to go home. On his way, he met other little kids. They asked if they could have some of the green apple. Tom said yes, and shared it with them.
>
> They all enjoyed the green apple together and enjoyed it very much. They thanked Tom for his search, and they said goodbye. Tom was happy to have found the green apple in the park, and he will never forget his adventure.

**Rating (item 11):** preferred = ____ (A / B / tie)  |  notes: 

## Item 12

**Prompt:** Write a short story using all of these words: better, decided, little.

**Story A**

> Once upon a time, there was a little bunny. He was very alert and always eager to try new things. One day, he saw a big carrot in a garden and decided to try to eat it. As he reached out to grab it, a fox appeared. The bunny was so scared that he ran away. But the fox was too fast for him. He chased after him, but the bunny was too fast for him. The bunny had to hide behind a bush until the fox went away. In the end, the bunny was safe and happy. He ate the carrot and was very alert.

**Story B**

> Once upon a time, there was a little bunny. He was always looking for something to do. One day, he decided to play a game of hide and seek. He found a perfect spot and hid behind a big tree. His friends were looking for him too.
>
> The bunny was very good at hiding. He closed his eyes and thought about what else he should do. Suddenly, he had a clever idea. He heard another sound. It was a bird! The bird asked the bunny to come closer.
>
> The bunny was scared, but he was also excited about the challenge. He looked at the bird and asked if he could be the one to win the game. The bird thought and thought. Finally, the bird said he should try something.
>
> The bunny was determined and so he took a deep breath and started to try it. He thought if he ran around the tree three times, he would win. After a few tries, he ended up so quickly that his friends couldn't catch up!
>
> The bird was about to give up when the bunny shouted out, "I did it!" The bird was so impressed by the bunny's determination and determination. He gave him a big hug.
>
> The bunny was so happy and grateful for

**Rating (item 12):** preferred = ____ (A / B / tie)  |  notes: 

## Item 13

**Prompt:** Write a short story using all of these words: dreamt, nightmare, please.

**Story A**

> Once upon a time, there was a little girl named Lily. She loved to play outside in the winter. One day, Lily and her friend Timmy were playing in the park. Lily had a thick stick that she used to draw pictures on the ground.
>
> Suddenly, Lily's mom called out, "Lily, it's time to go home!" Lily didn't want to leave the park, but she knew she had to listen to her mom. Lily and Timmy said goodbye and went home.
>
> Lily's mom was happy that she had fun in the park that day. She told Lily that she loved her very much and gave her a big hug. Lily smiled and felt happy that she had a good day. The end.

**Story B**

> Once upon a time, there was a little girl named Mia. She was a brave and adventurous girl who loved to dream big dreams.
>
> One day, Mia was dreaming of a big, beautiful world. She dreamed about swimming in the ocean and playing with all sorts of animals.
>
> â€œAh-ha!â€ she said to herself. â€œShr nightmares are always so dangerous.â€
>
> Suddenly, her dream changed. She was so happy that she woke up feeling cheerful.
>
> Miaâ€™s mother said, â€œHow was your dream?â€
>
> Mia replied, â€œIt was so much fun! I dreamed of flying like a bird in the sky.â€
>
> Miaâ€™s mother gave her a big hug and said, â€œThat sounds like a wonderful dream. Sweet dreams!â€
>
> Mia smiled and felt encouraged. She knew that her dreams were always there for her. She continued to dream big dreams every day.

**Rating (item 13):** preferred = ____ (A / B / tie)  |  notes: 

## Item 14

**Prompt:** Write a short story using all of these words: helping, look, loved.

**Story A**

> One day, a small girl named Lily was playing in the park. She had a red ball and a big pink bucket. Lily liked to play with her ball and the bucket. She felt powerful when she could throw the ball high in the air.
>
> Suddenly, a boy named Tom came to the park. He saw Lily and her ball. "Can I play with you?" he asked. Lily smiled and said, "Yes, let's play together!" They started to play with the ball and the bucket.
>
> Tom threw the ball high, and Lily caught it with the bucket. They played catch and laughed. After a while, they were tired. They sat under a tree and drank some water. Tom said, "I had fun today. Let's play again!" Lily nodded and said, "Yes, let's play again!"

**Story B**

> One day, a small girl named Amy was playing in the park. She had a bucket and a toy bucket in mind. In the park, she saw a butterfly. She was so happy to see the butterfly. She ran over to it and tried to catch it.
>
> But the butterfly was too fast. It flew away and disappeared. Amy was sad and started to cry.
>
> Suddenly, a man came over. He was very helpful. He asked Amy what was wrong. She told him about the butterfly and he said that it was too fast for her. He said that she should keep trying, so he went and gathered more buckets and put them in his truck.
>
> Amy smiled and gave him a big hug. She said goodbye and thanked him for being so helpful. Then she ran off to look for the butterfly.

**Rating (item 14):** preferred = ____ (A / B / tie)  |  notes: 

## Item 15

**Prompt:** Write a short story using all of these words: bulbs, hugged, wife.

**Story A**

> Once upon a time, there was a little girl named Lily. She loved to pick flowers in the garden with her mommy. One day, they saw a beautiful bulb that was very bright. "What is that, mommy?" asked Lily. "That's a flower bulb," said her mommy. "It will grow into a big, brilliant flower like the one we saw on the last day."
>
> Lily was very excited to see the flower grow. "Can we take it home?" she asked. "Of course, we can," said her mommy. They carefully picked the flower and brought it home.
>
> That night, they planted the bulb in their garden. A few days later, the flower grew into a big, brilliant flower. It was as if it was filled with all kinds of colors. "I'm so happy we planted that flower," said Lily. "Me too, me too," said her mommy with a big smile.

**Story B**

> Once upon a time, there was a little girl named Emma. Every day, Emma went to her rose garden to see if any bulbs were there. One day, she noticed a bulb in the corner. Emma thought it was so cool that she gave it a hug!
>
> The bulb seemed to smile back at her. Emma was so surprised by the bulb's hug that she hugged it even harder. When she hugged it, the bulb said "thank you". Emma was so proud of herself.
>
> The next day, Emma went to the garden again. She looked at the bulb and smiled. She was so happy and she hugged the bulb again. This time, she was able to find a big bulb in the same place.
>
> Emma was so excited. She gave the bulb a big hug of her own. She was so happy that she decided to give something else in her rose garden. She chose a bright flower and gave it to the bulb. The bulb smiled and thanked Emma.
>
> The bulb was so happy that Emma gave it a big hug. Emma was so proud that she hugged the bulb again. From that day on, Emma and the bulb were the best of friends.

**Rating (item 15):** preferred = ____ (A / B / tie)  |  notes: 

## Item 16

**Prompt:** Write a short story using all of these words: didn, longer, walking.

**Story A**

> Once upon a time, there was a little girl. She was excited to go on a walk with her Daddy. Her Daddy took her to the park and they had a great time.
>
> The little girl loved walking with Daddy. She would often look at the trees and the flowers and feel the sunshine on her face. Daddy would always smile when he saw her.
>
> On one of their walks, the little girl saw a big bird in the sky. She asked her Daddy, "What is that bird doing?" Her Daddy replied, "That is a big bird. It is helping the earth."
>
> The little girl was so excited to hear the bird's words. Daddy said, "You can help us take care of the earth and keep it safe." The girl smiled, and nodded.
>
> They continued their walk, knowing that their love and care was always appreciated. And they all lived happily ever after, taking lots of beautiful moments together.

**Story B**

> Once upon a time, there was a little girl. She was walking along the street and everything was going well until she suddenly stopped. This made her feel scared, so she looked around to see what was wrong.
>
> She noticed a big wall but no one was there to help her. She was feeling very impatient and wanted to find out why.
>
> So she bravely stepped on the wall, but it wouldn't budge. She looked around and noticed that the wall was stuck to the wall. She was so tired and frustrated.
>
> Suddenly, a big gust of wind came and the wall started to move. Suddenly, the wall started to move and the little girl was so excited! She jumped up and down with excitement.
>
> The wall led her anywhere she wanted to go and she was having a great time, until the wall started to shake. It was so strong that the little girl tripped over and the wall broke. The little girl felt so sad that she had broken the wall but she was still very brave.

**Rating (item 16):** preferred = ____ (A / B / tie)  |  notes: 

## Item 17

**Prompt:** Write a short story using all of these words: kind, named, share.

**Story A**

> Once upon a time, in a small house, lived a little boy named Tim. Tim had a big red ball. He loved to play with his ball every day.
>
> One day, Tim's ball rolled under a big tree. Tim tried to reach it, but he was too weak. He felt sad. Tim's friend, Sally, saw him and asked, "Tim, what's wrong?"
>
> Tim said, "My ball is under the big tree. I can't get it." Sally wanted to help Tim. Sally's mom came and helped Tim move the big tree. With a big push, the ball fell down. Tim was very happy.
>
> Tim said, "Thank you, Mom!" Now, Tim could play with his ball again. They played together all day long. Tim was not weak anymore. They had lots of fun and shared many happy days.

**Story B**

> Once upon a time, in a small house, lived a little girl named Lily. She had a big, pretty garden where she loved to play. In the garden, there were many flowers and trees. Lily liked to share her toys with her friends.
>
> One day, Lily found a small seed in her garden. She wanted to plant it and see what would grow. So, she planted the seed in the ground. She watered it every day and waited. Soon, a tiny green plant started to grow.
>
> Lily was so happy to see the little plant grow. She shared her toys with her friends, and they all played together in the garden. The kind girl learned that sharing is a good thing to do. And from that day on, Lily always shared her toys and felt kind to others.

**Rating (item 17):** preferred = ____ (A / B / tie)  |  notes: 

## Item 18

**Prompt:** Write a short story using all of these words: friends, hedge, idea.

**Story A**

> One day, John was walking in the park. Suddenly, he heard a noise coming from one of the bushes. He stopped to listen. He heard a low sound like someone was singing.
>
> John went closer to the bush to find out who was singing. He peeked inside and saw a little animal. It was very cute!
>
> John smiled and said, "Hello there! What are you singing?"
>
> The animal replied, "I'm singing a happy song. Do you want to join me?"
>
> John nodded and started to sing along with the animal. The animal sang so loud that it sounded like a song.
>
> They sang together until it was time to go. The animal said, "Thank you for singing with me, John. It was very fun."
>
> John smiled and said, "You're welcome! Come back soon and we can sing together again."
>
> The animal said goodbye and John went back to playing in the park. He was so happy that he had made a new friend.

**Story B**

> One day, John and Lucy were playing in the park. Lucy was very happy, but she was very sad because she had no friends. Suddenly, she saw a hedge that was very charming and colourful. She immediately ran up to it and said, "Hello hedge, what are you doing here?"
>
> The hedge replied, "I'm looking for some friends to play with. Don't you know that hedgees are not friendly?" Lucy thought for a moment and then said, "Maybe if you look in the wrong bushes you can find some of your friends."
>
> The hedge then nodded and said, "Very well. Just believe me and see what I can do. I'm sure I can find some friends to play with".
>
> So, Lucy and John went around the park looking for the hedge and soon enough they found a group of hedgey bunnies playing around. They were so happy to have found some new friends. They had lots of fun playing together in the park and soon enough Lucy was no longer sad.
>
> That day, Lucy and John had the most charming friendship that they had ever found.

**Rating (item 18):** preferred = ____ (A / B / tie)  |  notes: 

## Item 19

**Prompt:** Write a short story using all of these words: bottom, happy, swinging.

**Story A**

> One day, Jack and his mom went to the park. Jack saw a big tree and wanted to climb it. So his mom helped him up using the trunk and Jack kept going. He found a rope and held onto it tightly.
>
> Jack's mom smiled when she saw him. She said, "Jack, be careful. The rope is thin and the branches are thin." Jack nodded and said he would.
>
> Jack enjoyed climbing the tree, but he suddenly felt something sharp. He looked down and saw a sharp nail. He said, "Mom, I touched a nail!"
>
> His mom looked at the nail. She said, "That's very dangerous, Jack. We can't climb that tree. Come down from there."
>
> Jack put the rope down and walked away. He felt sad but he knew he made a mistake. He said, "I'm sorry, Mom. I won't climb that tree again."

**Story B**

> One day, Jack and his mom went to the park. Jack saw a big tree and wanted to swing on it. His mom said okay, so Jack ran up to the tree and started to swing. He went higher and higher and as he swung, he smiled and laughed.
>
> As he was swinging, he saw a bird in the tree. The bird was looking at Jack and his mom. Jack's mom said, "Let's go and swing from the tree, Jack." So Jack and his mom jumped from the tree and had lots of fun.
>
> After a while, Jack's mom said, "Let's go now." So they said goodbye to the bird, and Jack and his mom walked away. Jack kept swinging and singing, and the sun was shining brightly in the sky.
>
> Jack had a wonderful day at the park, swinging and singing, and he smiled a big smile. He said, "Mom, that was fun!"

**Rating (item 19):** preferred = ____ (A / B / tie)  |  notes: 

## Item 20

**Prompt:** Write a short story using all of these words: help, next, took.

**Story A**

> Once upon a time, there was a little girl who was very hungry. She wanted to help her Mom. So, she decided to take a walk in the park.
>
> When she got to the park she saw a lot of people. She asked her Mom why everyone was so hungry. Her Mom said that they needed food because they were all hungry.
>
> The little girl had a plan. She ran to the store and bought some yummy treats. She went back to the park and gave the treats to the people. They were so happy and thanked her.
>
> The little girl was happy that she could help. She was so proud of herself and smiled as she watched the people having a good time. She knew that her Mom would always be there to help.

**Story B**

> Once upon a time, there was a little girl who loved to explore the world. One day, she went for a walk along a path. She saw a little pond in the distance, and she decided to go take a closer look. As she got closer, she saw that the pond was deep and full of water.
>
> The little girl was so excited to discover this pond, she ran to the edge and looked inside. The water was so still and calm. She felt so happy to be there! She wondered what kinds of animals she could see.
>
> Suddenly, the little girl heard a voice coming from the pond. It was a little fish who was swimming around. The little girl smiled and waved at the fish. The little fish waved back and then swam away.
>
> The little girl felt so happy to have found the pond and to have seen the little fish swim around. She waved goodbye and went on her way, feeling content and calm.

**Rating (item 20):** preferred = ____ (A / B / tie)  |  notes: 

## Fill in

- Rater:
- Date:
- Items rated: /20
- Prefers A: , B: , tie:
- After unblinding, DPO preferred: , SFT preferred: , ties:
- Notes (failure examples welcome):
