import os
import csv


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.normpath(path)


def flattenjson(b, delim):
    val = {}
    for i in b.keys():
        if isinstance(b[i], dict):
            get = flattenjson(b[i], delim)
            for j in get.keys():
                val[i + delim + j] = get[j]
        else:
            val[i] = b[i]
    return val


def json_to_csv(output, output_path):
    input = [flattenjson(x, "__") for x in output]
    columns = [x for row in input for x in row.keys()]
    columns = list(set(columns))
    columns.sort(reverse=True)

    with open(output_path, "w") as out_file:
        csv_w = csv.writer(out_file)
        csv_w.writerow(columns)

        for i_r in input:
            csv_w.writerow(map(lambda x: i_r.get(x, ""), columns))
