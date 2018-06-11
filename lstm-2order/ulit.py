'''
def eval_ner(pred, gold):
    print 'Evaluating...'
    eval_dict = {}  # value=[#match, #pred, #gold]
    for p_1sent, g_1sent in zip(pred, gold):
        sent_len = len(p_1sent)
        i = 0
        while (i < sent_len):
            tp = p_1sent[i].split('-')
            tg = g_1sent[i].split('-')

            if len(tp) == 2:
                if tp[1] not in eval_dict:
                    eval_dict[tp[1]] = [0] * 3
                if tp[0] == 'B':
                    eval_dict[tp[1]][1] += 1
            if len(tg) == 2:
                if tg[1] not in eval_dict:
                    eval_dict[tg[1]] = [0] * 3
                if tg[0] == 'B':
                    eval_dict[tg[1]][2] += 1

            if p_1sent[i] == g_1sent[i] and tg[0] == 'B':
                name = tg[-1]
                i += 1

                if i < sent_len:
                    tp_bh = p_1sent[i].split('-')
                    tg_bh = g_1sent[i].split('-')
                    while (i < sent_len and p_1sent[i] == g_1sent[i] and p_1sent[i].split('-')[0] == 'I'):
                        i += 1
                    if i < sent_len:
                        tp_bh = p_1sent[i].split('-')
                        tg_bh = g_1sent[i].split('-')
                        if (g_1sent[i] == 'O' or tg_bh[0] == 'B') and (p_1sent[i] == 'O' or tp_bh[0] == 'B'):
                            eval_dict[name][0] += 1

                    else:
                        eval_dict[name][0] += 1

                else:
                    eval_dict[name][0] += 1

            else:
                i += 1

    agg_measure = [0.0] * 3
    agg_counts = [0] * 3
    for k, v in eval_dict.items():
        agg_counts = [sum(x) for x in zip(agg_counts, v)]
        prec = float(v[0]) / v[1] if v[1] != 0 else 0.0
        recall = float(v[0]) / v[2] if v[2] != 0 else 0.0
        F1 = 2 * prec * recall / (prec + recall) if prec != 0 and recall != 0 else 0.0
        agg_measure[0] += prec
        agg_measure[1] += recall
        agg_measure[2] += F1
        print k + ':', v[0], '\t', v[1], '\t', v[2], '\t', prec, '\t', recall, '\t', F1
    agg_measure = [v / len(eval_dict) for v in agg_measure]
    print 'Macro average:', '\t', agg_measure[0], '\t', agg_measure[1], '\t', agg_measure[2]
    prec = float(agg_counts[0]) / agg_counts[1] if agg_counts[1] != 0 else 0.0
    recall = float(agg_counts[0]) / agg_counts[2] if agg_counts[2] != 0 else 0.0
    F1 = 2 * prec * recall / (prec + recall) if prec != 0 and recall != 0 else 0.0
    print 'Micro average:', agg_counts[0], '\t', agg_counts[1], '\t', agg_counts[2], '\t', prec, '\t', recall, '\t', F1
    return {'p': prec, 'r': recall, 'f1': F1}
'''
def eval_ner(pred, gold):
    print "Evaluating"
    eval_dict = {}
    for p_sent, g_sent in zip(pred, gold):
        sent_len = len(p_sent)
        i = 0
        while(i < sent_len):
            tp = p_sent[i].split('-')
            tg = g_sent[i].split('-')
            if len(tp) == 2:
                if tp[1] not in eval_dict:
                    eval_dict[tp[1]] = [0]*3
                if tp[0] == 'B' :
                    eval_dict[tp[1]][1] += 1
            if len(tg) == 2:
                if tg[1] not in eval_dict:
                    eval_dict[tg[1]] = [0]*3 
                if tg[0] == 'B' :
                    eval_dict[tg[1]][2] += 1
            if p_sent[i] == g_sent[i] and tg[0] == 'B':
                i += 1
                while(i < sent_len):
                    if (p_sent[i][0] == 'B' or p_sent[i][0] == 'O') and (g_sent[i][0] == 'B' or g_sent[i][0] == 'O'):
                        eval_dict[tg[1]][0] += 1
                        i -= 1
                        break
                    elif (p_sent[i][0] == 'I' and g_sent[i][0] == 'I'):
                        i += 1
                    else:
                        i -= 1
                        break
                
                if i == sent_len:
                    eval_dict[tg[1]][0] += 1
                    break
            i += 1

    agg_measure = [0.0]*3
    agg_counts = [0]*3
    for k, v in eval_dict.items():
        agg_counts = [sum(x) for x in zip(agg_counts, v)]
        prec = float(v[0])/v[1] if v[1] != 0 else 0.0 
        recall = float(v[0])/v[2] if v[2] != 0 else 0.0
        F1 = 2*prec*recall/(prec+recall) if prec != 0 and recall != 0 else 0.0
        agg_measure[0] += prec
        agg_measure[1] += recall
        agg_measure[2] += F1
        print k+':', v[0], '\t', v[1], '\t', v[2], '\t', prec, '\t', recall, '\t', F1
    agg_measure = [v/len(eval_dict) for v in agg_measure]
    print 'Macro average:', '\t', agg_measure[0], '\t', agg_measure[1], '\t', agg_measure[2]
    prec = float(agg_counts[0])/agg_counts[1] if agg_counts[1] != 0 else 0.0
    recall = float(agg_counts[0])/agg_counts[2] if agg_counts[2] != 0 else 0.0
    F1 = 2*prec*recall/(prec+recall) if prec != 0 and recall != 0 else 0.0
    print 'Micro average:', agg_counts[0], '\t', agg_counts[1], '\t', agg_counts[2], '\t', prec, '\t', recall, '\t', F1 
    return {'p': prec, 'r': recall, 'f1': F1}